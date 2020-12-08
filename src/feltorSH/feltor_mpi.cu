#include <iostream>
#include <iomanip>
#include <vector>
#include <sstream>
#include <cmath>
// #define DG_DEBUG

#include <mpi.h> //activate mpi

#include "netcdf_par.h"

#include "dg/algorithm.h"
#include "dg/file/file.h"

#include "feltor.cuh"
#include "parameters.h"


int main( int argc, char* argv[])
{
     ////////////////////////////////setup MPI///////////////////////////////
    int provided;
    MPI_Init_thread( &argc, &argv, MPI_THREAD_FUNNELED, &provided);
    if( provided != MPI_THREAD_FUNNELED)
    {
        std::cerr << "wrong mpi-thread environment provided!\n";
        return -1;
    }
    int rank, size;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank);
    MPI_Comm_size( MPI_COMM_WORLD, &size);
    ////////////////////////Parameter initialisation//////////////////////////
    Json::Value js;
    Json::CharReaderBuilder parser;
    parser["collectComments"] = false;
    std::string errs;
    if( argc != 3)
    {
        if(rank==0)std::cerr << "ERROR: Wrong number of arguments!\nUsage: "<< argv[0]<<" [inputfile] [outputfile]\n";
        return -1;
    }
    else 
        file::file2Json( argv[1], js, file::comments::are_forbidden);
    std::string input = js.toStyledString(); //save input without comments, which is important if netcdf file is later read by another parser
    const eule::Parameters p( js);
    if(rank==0)p.display( std::cout);
     ////////////////////////////////setup MPI///////////////////////////////
    int periods[2] = {false, false}; //non-, non-, periodic

    if( p.bc_x == dg::PER) periods[0] = true;
    if( p.bc_y == dg::PER) periods[1] = true;
    int np[2];
    if(rank==0)
    {
        std::cin>> np[0] >> np[1] ;
        std::cout << "Computing with "<<np[0]<<" x "<<np[1] << " = "<<size<<std::endl;
        assert( size == np[0]*np[1]);
    }
    MPI_Bcast( np, 2, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Comm comm;
    MPI_Cart_create( MPI_COMM_WORLD, 2, np, periods, true, &comm);
    //////////////////////////////////////////////////////////////

     //Make grid
    dg::MPIGrid2d grid( 0., p.lx, 0.,p.ly, p.n, p.Nx, p.Ny, p.bc_x, p.bc_y,comm);
    dg::MPIGrid2d grid_out( 0., p.lx, 0.,p.ly, p.n_out, p.Nx_out, p.Ny_out, p.bc_x, p.bc_y, comm);  
    //create RHS 
    if(rank==0) std::cout << "Constructing Explicit...\n";
    eule::Explicit<dg::CartesianMPIGrid2d, dg::MDMatrix, dg::MDVec > feltor( grid, p); //initialize before rolkar!
    if(rank==0) std::cout << "Constructing Implicit...\n";
    eule::Implicit<dg::CartesianMPIGrid2d, dg::MDMatrix, dg::MDVec > rolkar( grid, p);
    if(rank==0) std::cout << "Done!\n";

    /////////////////////The initial field///////////////////////////////////////////
       //initial perturbation
    dg::Gaussian init0( p.posX*p.lx, p.posY*p.ly, p.sigma, p.sigma, p.amp);
    dg::CONSTANT prof(p.bgprofamp );
    std::vector<dg::MDVec> y0(4, dg::evaluate( prof, grid)), y1(y0); //Ne,Ni,Te,Ti = prof    
   
   //initialization via N_i,T_I ->n_e, t_i=t_e
    y1[1] = dg::evaluate( init0, grid);
    dg::blas1::pointwiseDot(y1[1], y0[1],y1[1]); //<n>*ntilde    
    dg::blas1::axpby( 1., y1[1], 1., y0[1]); //initialize Ni = <n> + <n>*ntilde
    if (p.iso == 1) dg::blas1::axpby( 1.,y1[2], 0., y0[3]); //initialize Ti = prof
    if (p.iso == 0) dg::blas1::axpby( 1.,y0[1], 0., y0[3]); //initialize Ti = N_i
    dg::blas1::transform(y0[1], y0[1], dg::PLUS<>(-(p.bgprofamp + p.nprofileamp))); //= Ni - bg
    if(rank==0) std::cout << "intiialize ne" << std::endl;
    if( p.init == 0)
        feltor.initializene( y0[1],y0[3], y0[0]);    //ne -bg
    else  
        dg::blas1::axpby( 1., y0[1], 0., y0[0], y0[0]); // for Omega*=0
    if(rank==0) std::cout << "Done!\n";    
    
    if(rank==0) std::cout << "intialize ti=te" << std::endl;
    if (p.iso == 1) {
        dg::blas1::transform(y0[3], y0[3], dg::PLUS<>(-(p.bgprofamp + p.nprofileamp))); // =Ti - bg
        dg::blas1::axpby( 1.,y0[3], 0., y0[2]); //initialize Ti = N_i
    }
    if (p.iso == 0) {
        dg::blas1::transform(y0[1], y0[1], dg::PLUS<>(+(p.bgprofamp + p.nprofileamp))); //Ni
        dg::blas1::pointwiseDot(y0[1],y0[3],y1[3]); // = Ni Ti
        dg::blas1::transform(y1[3], y1[3], dg::PLUS<>(-(p.bgprofamp + p.nprofileamp)*(p.bgprofamp + p.nprofileamp))); //Pi = Pi - bg^2

        feltor.initializepi(y1[3],y0[3], y0[2]); // = pi-bg^2    
        //compute ti-bg = ((pi-bg^2) +bg^2)/ne -bg
        dg::blas1::transform(y0[2], y0[2], dg::PLUS<>(+(p.bgprofamp + p.nprofileamp)*(p.bgprofamp + p.nprofileamp)));
        dg::blas1::transform(y0[0], y0[0], dg::PLUS<>(+(p.bgprofamp + p.nprofileamp))); //=ne    
        dg::blas1::pointwiseDivide(y0[2],y0[0],y0[2]);

        dg::blas1::transform(y0[2], y0[2], dg::PLUS<>(-(p.bgprofamp + p.nprofileamp)));
        dg::blas1::transform(y0[0], y0[0], dg::PLUS<>(-(p.bgprofamp + p.nprofileamp))); // =ne-bg
        dg::blas1::transform(y0[3], y0[3], dg::PLUS<>(-(p.bgprofamp + p.nprofileamp))); // =Ti - bg
        dg::blas1::transform(y0[1], y0[1], dg::PLUS<>(-(p.bgprofamp + p.nprofileamp))); // =Ni - bg 
    }
    if(rank==0) std::cout << "Done!\n";

    
    dg::Karniadakis< std::vector<dg::MDVec> > karniadakis( y0, y0[0].size(), p.eps_time);
    if(rank==0) std::cout << "intiialize Timestepper" << std::endl;
    karniadakis.init( feltor, rolkar, 0., y0, p.dt);
    if(rank==0) std::cout << "Done!\n";    
    /////////////////////////////set up netcdf/////////////////////////////////////
    file::NC_Error_Handle err;
    int ncid;
    MPI_Info info = MPI_INFO_NULL;
//     err = nc_create( argv[2],NC_NETCDF4|NC_CLOBBER, &ncid);//MPI OFF
    err = nc_create_par( argv[2], NC_NETCDF4|NC_MPIIO|NC_CLOBBER, comm, info, &ncid); //MPI ON
    err = nc_put_att_text( ncid, NC_GLOBAL, "inputfile", input.size(), input.data());
    //const int version[3] = {FELTOR_MAJOR_VERSION, FELTOR_MINOR_VERSION, FELTOR_SUBMINOR_VERSION};
    //err = nc_put_att_int( ncid, NC_GLOBAL, "feltor_major_version", NC_INT, 1, &version[0]);
    //err = nc_put_att_int( ncid, NC_GLOBAL, "feltor_minor_version", NC_INT, 1, &version[1]);
    //err = nc_put_att_int( ncid, NC_GLOBAL, "feltor_subminor_version", NC_INT, 1, &version[2]);
    int dim_ids[3], tvarID;
    dg::Grid2d global_grid_out ( 0., p.lx, 0.,p.ly, p.n_out, p.Nx_out, p.Ny_out, p.bc_x, p.bc_y);  
    err = file::define_dimensions( ncid, dim_ids, &tvarID, global_grid_out);
    err = nc_enddef( ncid);
    err = nc_redef(ncid);

    //field IDs
    std::string names[6] = {"electrons", "ions", "Telectrons","Tions", "potential","vor"}; 
    int dataIDs[6]; 
    for( unsigned i=0; i<6; i++){
        err = nc_def_var( ncid, names[i].data(), NC_DOUBLE, 3, dim_ids, &dataIDs[i]);
        err = nc_var_par_access( ncid, dataIDs[i], NC_COLLECTIVE);
    }
    err = nc_var_par_access( ncid, tvarID, NC_COLLECTIVE);
    //energy IDs
    int EtimeID, EtimevarID;
    err = file::define_time( ncid, "energy_time", &EtimeID, &EtimevarID);
    err = nc_var_par_access( ncid, EtimevarID, NC_COLLECTIVE);

    int energyID, massID, energyIDs[3], dissID, dEdtID, accuracyID;
    err = nc_def_var( ncid, "energy",   NC_DOUBLE, 1, &EtimeID, &energyID);
    err = nc_var_par_access( ncid, energyID, NC_COLLECTIVE);
    err = nc_def_var( ncid, "mass",   NC_DOUBLE, 1, &EtimeID, &massID);
    err = nc_var_par_access( ncid, massID, NC_COLLECTIVE);

    std::string energies[3] = {"Se", "Si", "Uperp"}; 
    for( unsigned i=0; i<3; i++){
        err = nc_def_var( ncid, energies[i].data(), NC_DOUBLE, 1, &EtimeID, &energyIDs[i]);
        err = nc_var_par_access( ncid, energyIDs[i], NC_COLLECTIVE);
    }
    err = nc_def_var( ncid, "dissipation",   NC_DOUBLE, 1, &EtimeID, &dissID);
    err = nc_var_par_access( ncid, dissID, NC_COLLECTIVE);
    err = nc_def_var( ncid, "dEdt",     NC_DOUBLE, 1, &EtimeID, &dEdtID);
    err = nc_var_par_access( ncid, dEdtID, NC_COLLECTIVE);
    err = nc_def_var( ncid, "accuracy", NC_DOUBLE, 1, &EtimeID, &accuracyID);
    err = nc_var_par_access( ncid, accuracyID, NC_COLLECTIVE);
    err = nc_enddef(ncid);
    ///////////////////////////////////first output/////////////////////////
    if(rank==0) std::cout << "First output ... \n";
    int dims[2],  coords[2];
    MPI_Cart_get( comm, 2, dims, periods, coords);
    size_t count[3] = {1, grid_out.n()*grid_out.local().Ny(), grid_out.n()*grid_out.local().Nx()};
    size_t start[3] = {0, coords[1]*count[1],          coords[0]*count[2]}; 
    dg::MDVec transfer( dg::evaluate(dg::zero, grid));
    dg::DVec transferD( dg::evaluate(dg::zero, grid_out.local()));
    dg::HVec transferH( dg::evaluate(dg::zero, grid_out.local()));
    dg::IDMatrix interpolate = dg::create::interpolation( grid_out.local(), grid.local()); //create local interpolation matrix

    for( unsigned i=0; i<4; i++)
    {
        dg::blas2::gemv( interpolate, y0[i].data(), transferD);
        dg::blas1::transfer( transferD, transferH);
        err = nc_put_vara_double( ncid, dataIDs[i], start, count, transferH.data() );
    }
    //pot
    transfer = feltor.potential()[0];
    dg::blas2::gemv( interpolate, transfer.data(), transferD);
    dg::blas1::transfer( transferD, transferH);
    err = nc_put_vara_double( ncid, dataIDs[4], start, count, transferH.data() );
    //Vor
    transfer = feltor.potential()[0];
    dg::blas2::gemv( rolkar.laplacianM(), transfer, y1[1]);        
    dg::blas2::gemv( interpolate,y1[1].data(), transferD);
    dg::blas1::transfer( transferD, transferH);
    err = nc_put_vara_double( ncid, dataIDs[5], start, count, transferH.data() );
    double time = 0;

    err = nc_put_vara_double( ncid, tvarID, start, count, &time);
    err = nc_put_vara_double( ncid, EtimevarID, start, count, &time);

    size_t Estart[] = {0};
    size_t Ecount[] = {1};
    double energy0 = feltor.energy(), mass0 = feltor.mass(), E0 = energy0, mass = mass0, E1 = 0.0, dEdt = 0., diss = 0., accuracy=0.;


    std::vector<double> evec = feltor.energy_vector();
    err = nc_put_vara_double( ncid, energyID, Estart, Ecount, &energy0);
    err = nc_put_vara_double( ncid, massID,   Estart, Ecount, &mass0);
    for( unsigned i=0; i<3; i++)
        err = nc_put_vara_double( ncid, energyIDs[i], Estart, Ecount, &evec[i]);

    err = nc_put_vara_double( ncid, dissID,     Estart, Ecount,&diss);
    err = nc_put_vara_double( ncid, dEdtID,     Estart, Ecount,&dEdt);
    err = nc_put_vara_double( ncid, accuracyID, Estart, Ecount,&accuracy);    
//     err = nc_close(ncid);
    if(rank==0) std::cout << "First write successful!\n";

    ///////////////////////////////////////Timeloop/////////////////////////////////
#ifdef DG_BENCHMARK
    dg::Timer t;
    t.tic();
#endif //DG_BENCHMARK
    unsigned step = 0;
    for( unsigned i=1; i<=p.maxout; i++)
    {

#ifdef DG_BENCHMARK
        dg::Timer ti;
        ti.tic();
#endif//DG_BENCHMARK
        for( unsigned j=0; j<p.itstp; j++)
        {
            try{ karniadakis.step( feltor, rolkar, time, y0);}
            catch( dg::Fail& fail) { 
                if(rank==0) std::cerr << "CG failed to converge to "<<fail.epsilon()<<"\n";
                if(rank==0) std::cerr << "Does Simulation respect CFL condition?\n";
                err = nc_close(ncid);
                MPI_Finalize();
                return -1;
            }
            step++;
            Estart[0] = step;
            E1 = feltor.energy(), mass = feltor.mass(), diss = feltor.energy_diffusion();
            dEdt = (E1 - E0)/p.dt; 
            E0 = E1;
            accuracy = 2.*fabs( (dEdt-diss)/(dEdt + diss));
            evec = feltor.energy_vector();
//             err = nc_open(argv[2], NC_WRITE, &ncid);
            err = nc_put_vara_double( ncid, EtimevarID, Estart, Ecount, &time);
            err = nc_put_vara_double( ncid, energyID, Estart, Ecount, &E1);
            err = nc_put_vara_double( ncid, massID,   Estart, Ecount, &mass);
            for( unsigned i=0; i<3; i++)
            {

                err = nc_put_vara_double( ncid, energyIDs[i], Estart, Ecount, &evec[i]);
            }
            err = nc_put_vara_double( ncid, dissID,     Estart, Ecount,&diss);
            err = nc_put_vara_double( ncid, dEdtID,     Estart, Ecount,&dEdt);
           
            err = nc_put_vara_double( ncid, accuracyID, Estart, Ecount,&accuracy);
            if(rank==0) std::cout << "(m_tot-m_0)/m_0: "<< (feltor.mass()-mass0)/mass0<<"\t";
            if(rank==0) std::cout << "(E_tot-E_0)/E_0: "<< (E1-energy0)/energy0<<"\t";
            if(rank==0) std::cout <<" d E/dt = " << dEdt <<" Lambda = " << diss << " -> Accuracy: "<< accuracy << "\n";
//             err = nc_close(ncid);
 
        }
#ifdef DG_BENCHMARK
        ti.toc();
        if(rank==0)std::cout << "\n\t Step "<<step <<" of "<<p.itstp*p.maxout <<" at time "<<time;
        if(rank==0) std::cout << "\n\t Average time for one step: "<<ti.diff()/(double)p.itstp<<"s\n\n"<<std::flush;
        ti.tic();
#endif//DG_BENCHMARK
        //////////////////////////write fields////////////////////////
        start[0] = i;
//         err = nc_open(argv[2], NC_WRITE, &ncid);
        for( unsigned j=0; j<4; j++)
        {
            dg::blas2::gemv( interpolate, y0[j].data(), transferD);
            dg::blas1::transfer( transferD, transferH);
            err = nc_put_vara_double( ncid, dataIDs[j], start, count, transferH.data());
        }
        transfer = feltor.potential()[0];
        dg::blas2::gemv( interpolate, transfer.data(), transferD);
        dg::blas1::transfer( transferD, transferH);
        err = nc_put_vara_double( ncid, dataIDs[4], start, count, transferH.data() );
        transfer = feltor.potential()[0];
        dg::blas2::gemv( rolkar.laplacianM(), transfer, y1[1]);        //correct?    
        dg::blas2::gemv( interpolate,y1[1].data(), transferD);
        dg::blas1::transfer( transferD, transferH);
        err = nc_put_vara_double( ncid, dataIDs[5], start, count, transferH.data() );
        err = nc_put_vara_double( ncid, tvarID, start, count, &time);
//         err = nc_close(ncid);
#ifdef DG_BENCHMARK
        ti.toc();
        if(rank==0)std::cout << "\n\t Time for output: "<<ti.diff()<<"s\n\n"<<std::flush;
#endif//DG_BENCHMARK
    }
#ifdef DG_BENCHMARK
    t.toc(); 
    unsigned hour = (unsigned)floor(t.diff()/3600);
    unsigned minute = (unsigned)floor( (t.diff() - hour*3600)/60);
    double second = t.diff() - hour*3600 - minute*60;
    if(rank==0) std::cout << std::fixed << std::setprecision(2) <<std::setfill('0');
    if(rank==0) std::cout <<"Computation Time \t"<<hour<<":"<<std::setw(2)<<minute<<":"<<second<<"\n";
    if(rank==0) std::cout <<"which is         \t"<<t.diff()/p.itstp/p.maxout<<"s/step\n";
#endif //DG_BENCHMARK
    err = nc_close(ncid);
    MPI_Finalize();
    return 0;

}

