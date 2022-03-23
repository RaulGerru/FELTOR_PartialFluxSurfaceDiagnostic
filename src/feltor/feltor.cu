#include <iostream>
#include <iomanip>
#include <vector>
#include <map>
#include <sstream>
#include <cmath>
#include <csignal>

#ifdef WITH_MPI
#include <mpi.h>
#endif //WITH_MPI

#ifndef WITHOUT_GLFW
#include "draw/host_window.h"
#endif // WITHOUT_GLFW

#include "dg/file/file.h"
#include "feltor.h"
#include "init.h"
#include "feltordiag_conserv.h"
#include "init_from_file.h"

#ifdef WITH_MPI
//ATTENTION: in slurm should be used with --signal=SIGINT@30 (<signal>@<time in seconds>)
void sigterm_handler(int signal)
{
    int rank;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank);
    std::cout << " pid "<<rank<<" sigterm_handler, got signal " << signal << std::endl;
    MPI_Finalize();
    exit(signal);
}
#endif //WITH_MPI

int main( int argc, char* argv[])
{
#ifdef WITH_MPI
    ////////////////////////////////setup MPI///////////////////////////////
    dg::mpi_init( argc, argv);
    MPI_Comm comm;
    dg::mpi_init3d( dg::DIR, dg::DIR, dg::PER, comm, std::cin, true);
    int rank;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank);

    std::signal(SIGINT, sigterm_handler);
    std::signal(SIGTERM, sigterm_handler);
#endif //WITH_MPI
    ////////////////////////Parameter initialisation//////////////////////////
    dg::file::WrappedJsonValue js( dg::file::error::is_throw);
    if( argc != 2 && argc != 3 && argc != 4)
    {
        DG_RANK0 std::cerr << "ERROR: Wrong number of arguments!\nUsage: "
                << argv[0]<<" [input.json] \n OR \n"
                << argv[0]<<" [input.json] [output.nc]\n OR \n"
                << argv[0]<<" [input.json] [output.nc] [initial.nc] "<<std::endl;
        dg::abort_program();
    }
    try{
        dg::file::file2Json( argv[1], js.asJson(),
                dg::file::comments::are_discarded, dg::file::error::is_throw);
        feltor::Parameters p( js);
    } catch( std::exception& e) {
        DG_RANK0 std::cerr << "ERROR in input file "<<argv[1]<<std::endl;
        DG_RANK0 std::cerr << e.what()<<std::endl;
        dg::abort_program();
    }
    std::string geometry_params = js["magnetic_field"]["input"].asString();
    if( geometry_params == "file")
    {
        std::string path = js["magnetic_field"]["file"].asString();
        try{
            dg::file::file2Json( path, js.asJson()["magnetic_field"]["params"],
                    dg::file::comments::are_discarded, dg::file::error::is_throw);
        }catch(std::runtime_error& e)
        {
            DG_RANK0 std::cerr << "ERROR in geometry file "<<path<<std::endl;
            DG_RANK0 std::cerr << e.what()<<std::endl;
            dg::abort_program();
        }
    }
    else if( geometry_params != "params")
    {
        DG_RANK0 std::cerr << "Error: Unknown magnetic field input '"
                           << geometry_params<<"'. Exit now!\n";
        dg::abort_program();
    }
    const feltor::Parameters p( js);
    DG_RANK0 std::cout << js.asJson() <<  std::endl;
    std::string inputfile = js.asJson().toStyledString();
    dg::geo::TokamakMagneticField mag, mod_mag, unmod_mag;
    dg::geo::CylindricalFunctor wall, transition, sheath, sheath_coordinate =
        [](double x, double y){return 0.;};
    try{
        mag = unmod_mag = dg::geo::createMagneticField(js["magnetic_field"]["params"]);
        mod_mag = dg::geo::createModifiedField(js["magnetic_field"]["params"],
                js["boundary"]["wall"], wall, transition);
    }catch(std::runtime_error& e)
    {
        DG_RANK0 std::cerr << "ERROR in input file "<<argv[1]<<std::endl;
        DG_RANK0 std::cerr <<e.what()<<std::endl;
        dg::abort_program();
    }

    ////////////////////////////////set up computations///////////////////////////
#ifdef WITH_MPI
    int dims[3], periods[3], coords[3];
    MPI_Cart_get( comm, 3, dims, periods, coords);
    if( dims[2] >= (int)p.Nz)
    {
        DG_RANK0 std::cerr << "ERROR: Number of processes in z "<<dims[2]
                    <<" may not be larger or equal Nz "<<p.Nz<<std::endl;
        dg::abort_program();
    }
#endif //WITH_MPI
    //Make grids
    const double Rmin=mag.R0()-p.boxscaleRm*mag.params().a();
    const double Zmin=-p.boxscaleZm*mag.params().a();
    const double Rmax=mag.R0()+p.boxscaleRp*mag.params().a();
    const double Zmax=p.boxscaleZp*mag.params().a();
    dg::x::CylindricalGrid3d grid( Rmin, Rmax, Zmin, Zmax, 0, 2.*M_PI,
        p.n, p.Nx, p.Ny, p.symmetric ? 1 : p.Nz, p.bcxN, p.bcyN, dg::PER
        #ifdef WITH_MPI
        , comm
        #endif //WITH_MPI
        );

    if( p.periodify)
    {
        unmod_mag = dg::geo::periodify( unmod_mag, Rmin, Rmax, Zmin, Zmax, dg::NEU, dg::NEU);
        mod_mag = dg::geo::periodify( mod_mag, Rmin, Rmax, Zmin, Zmax, dg::NEU, dg::NEU);
    }
    if( p.modify_B)
        mag = mod_mag;
    else
        mag = unmod_mag;

    DG_RANK0 std::cout << "# Constructing Feltor...\n";
    feltor::Explicit< dg::x::CylindricalGrid3d, dg::x::IDMatrix,
        dg::x::DMatrix, dg::x::DVec> feltor( grid, p, mag, js);
    feltor::Implicit< dg::x::CylindricalGrid3d, dg::x::IDMatrix,
        dg::x::DMatrix, dg::x::DVec> implicit( feltor);
    DG_RANK0 std::cout << "# Done!\n";

    feltor.set_wall( p.wall_rate, dg::construct<dg::x::DVec>( dg::pullback(
                    wall, grid)), p.nwall, p.uwall );
    dg::Timer t;
    if( p.sheath_bc != "none")
    {
        t.tic();
        DG_RANK0 std::cout << "# Compute Sheath coordinates \n";
        try{
            dg::Grid2d sheath_walls( Rmin, Rmax, Zmin, Zmax, 1, 1, 1);
            dg::geo::createSheathRegion( js["boundary"]["sheath"],
                dg::geo::createMagneticField(js["magnetic_field"]["params"]),
                wall, sheath_walls, sheath);
            // sheath is created on feltor magnetic field
            sheath_coordinate = dg::geo::WallFieldlineCoordinate(
                    dg::geo::createBHat( mag), sheath_walls,
                    p.sheath_max_angle, 1e-6, p.sheath_coord);
        }catch(std::runtime_error& e)
        {
            DG_RANK0 std::cerr << "ERROR in input file "<<argv[1]<<std::endl;
            DG_RANK0 std::cerr <<e.what()<<std::endl;
            dg::abort_program();
        }
        dg::x::HVec coord2d = dg::pullback( sheath_coordinate, *grid.perp_grid());
        dg::x::DVec coord3d;
        dg::assign3dfrom2d( coord2d, coord3d, grid);
        feltor.set_sheath(
                p.sheath_rate,
                dg::construct<dg::x::DVec>(dg::pullback( sheath, grid)),
                coord3d);
        t.toc();
        DG_RANK0 std::cout << "# ... took  "<<t.diff()<<"s\n";
    }

    DG_RANK0 std::cout << "# Set Source \n";
    try{
        bool fixed_profile;
        dg::x::HVec ne_profile, source_profile;
        double minne = 0., minrate = 0., minalpha = 0.;
        source_profile = feltor::source_profiles( feltor,
            fixed_profile, ne_profile, grid, mag, unmod_mag, js["source"],
            minne, minrate, minalpha);
        feltor.set_source( fixed_profile,
                dg::construct<dg::x::DVec>(ne_profile), p.source_rate,
                dg::construct<dg::x::DVec>(source_profile),
                minne, minrate, minalpha);
    }catch ( std::out_of_range& error){
        DG_RANK0 std::cerr << "ERROR: in source: "<<error.what();
        DG_RANK0 std::cerr <<"Is there a spelling error? I assume you do not want to continue with the wrong source so I exit! Bye Bye :)"<<std::endl;
        dg::abort_program();
    }
    /// /////////////The initial field//////////////////////////////////////////
    double time = 0.;
    std::array<std::array<dg::x::DVec,2>,2> y0;
    dg::geo::Nablas<dg::x::CylindricalGrid3d, dg::x::DVec, dg::x::DMatrix> nabla(grid, p, mag);
    std::array<dg::x::DVec, 3> gradPsip;
    gradPsip[0] =  dg::evaluate( mag.psipR(), grid);
    gradPsip[1] =  dg::evaluate( mag.psipZ(), grid);
    gradPsip[2] =  dg::evaluate( dg::zero, grid); //zero
    feltor::Variables var{
        feltor, y0, p, mag, nabla, gradPsip, gradPsip, gradPsip, gradPsip,
        dg::construct<dg::x::DVec>( dg::pullback( dg::geo::Hoo(mag),grid)),
        0., // duration
        0 // nfailed
    };
    DG_RANK0 std::cout << "# Set Initial conditions ... \n";
    t.tic();
    if( argc == 4 )
    {
        try{
            y0 = feltor::init_from_file(argv[3], grid, p, time);
        }catch (std::exception& e){
            DG_RANK0 std::cerr << "ERROR occured initializing from file "<<argv[3]<<std::endl;
            DG_RANK0 std::cerr << e.what()<<std::endl;
            dg::abort_program();
        }
    }
    else
    {
        try{
            y0 = feltor::initial_conditions(feltor, grid, p, mag, unmod_mag,
                    js["init"], js["magnetic_field"], time, sheath_coordinate );
#ifdef WITH_NAVIER_STOKES
            std::string advection = js["advection"].get("type", "velocity-staggered").asString();
            if( advection == "log-staggered")
                dg::blas1::transform( y0[0], y0[0], dg::LN<double>());
            else if( advection == "staggered")
            {
                // MW: Not correct, density is not staggered,only
                // works when velocity is zero
                dg::blas1::pointwiseDot( y0[0], y0[1], y0[1]);
            }
#endif // WITH_NAVIER_STOKES
        }catch ( dg::Error& error){
            DG_RANK0 std::cerr << error.what();
            DG_RANK0 std::cerr << "Is there a spelling error? I assume you do not want to continue with the wrong parameter so I exit! Bye Bye :)\n";
            dg::abort_program();
        }
    }
    t.toc();
    DG_RANK0 std::cout << "# ... took  "<<t.diff()<<"s\n";


    ///////////////////////////////////////////////////////////////////////////
    DG_RANK0 std::cout << "# Initialize Timestepper" << std::endl;
    dg::ExplicitMultistep< std::array<std::array<dg::x::DVec,2>,2>> multistep;
    feltor::ImplicitSolver<dg::x::CylindricalGrid3d, dg::x::IDMatrix, dg::x::DMatrix, dg::x::DVec> solver;
    dg::ImExMultistep< std::array<std::array<dg::x::DVec,2>,2> > multistep_imex;
    dg::Adaptive< dg::ERKStep< std::array<std::array<dg::x::DVec,2>,2>>> adapt;
    dg::Adaptive< dg::ARKStep< std::array<std::array<dg::x::DVec,2>,2>>> adapt_ark;
    double rtol = 0., atol = 0., dt = 0., reject_limit = 2;
    if( p.timestepper == "multistep")
    {
        multistep = { p.tableau, y0};
        dt = js[ "timestepper"]["dt"].asDouble( 0.01);
    }
    else if( p.timestepper == "multistep-imex")
    {
        double eps_time = js[ "timestepper"]["solver"].get("eps_time", 1e-10).asDouble();
        solver = { feltor, eps_time};
        multistep_imex = { p.tableau, y0};
        dt = js[ "timestepper"]["dt"].asDouble( 0.01);
    }
    else if (p.timestepper == "adaptive")
    {
        adapt = {p.tableau, y0};
        adapt.stepper().ignore_fsal();
        rtol = js[ "timestepper"][ "rtol"].asDouble( 1e-7);
        atol = js[ "timestepper"][ "atol"].asDouble( 1e-10);
        dt = 1e-5; //that should be a small enough initial guess
        reject_limit = js["timestepper"].get("reject-limit", 2).asDouble();
    }
    else if (p.timestepper == "adaptive-imex")
    {
        double eps_time = js[ "timestepper"]["solver"].get("eps_time", 1e-10).asDouble();
        solver = { feltor, eps_time};
        adapt_ark = { p.tableau, y0};
        rtol = js[ "timestepper"][ "rtol"].asDouble( 1e-7);
        atol = js[ "timestepper"][ "atol"].asDouble( 1e-10);
        dt = 1e-5; //that should be a small enough initial guess
        reject_limit = js["timestepper"].get("reject-limit", 2).asDouble();
    }
    else
    {
        DG_RANK0 std::cerr << "Error: Unrecognized timestepper: '"
                           << p.timestepper << "'! Exit now!";
        dg::abort_program();
    }
    DG_RANK0 std::cout << "Done!\n";

    /// //////////////////////////set up netcdf/////////////////////////////////////
    if( p.output == "netcdf")
    {
        // helper variables for output computations
        unsigned cx = js["output"]["compression"].get(0u,1).asUInt();
        unsigned cy = js["output"]["compression"].get(1u,1).asUInt();
        unsigned n_out = p.n, Nx_out = p.Nx/cx, Ny_out = p.Ny/cy, Nz_out = p.Nz;
        dg::x::CylindricalGrid3d g3d_out( Rmin, Rmax, Zmin, Zmax, 0, 2.*M_PI,
            n_out, Nx_out, Ny_out, p.symmetric ? 1 : Nz_out, p.bcxN, p.bcyN, dg::PER
            #ifdef WITH_MPI
            , comm
            #endif //WITH_MPI
            );
        std::unique_ptr<typename dg::x::CylindricalGrid3d::perpendicular_grid>
            g2d_out_ptr  ( dynamic_cast<typename
                    dg::x::CylindricalGrid3d::perpendicular_grid*>(
                        g3d_out.perp_grid()));
#ifdef WITH_MPI
        unsigned local_size2d = g2d_out_ptr->local().size();
#else
        unsigned local_size2d = g2d_out_ptr->size();
#endif
        std::map<std::string, dg::Simpsons<dg::x::HVec>> time_integrals;
        dg::Average<dg::x::HVec> toroidal_average( g3d_out, dg::coo3d::z, "simple");
        dg::MultiMatrix<dg::x::HMatrix,dg::x::HVec> projectH =
            dg::create::fast_projection( grid, 1, cx, cy);
        dg::MultiMatrix<dg::x::DMatrix,dg::x::DVec> projectD =
            dg::create::fast_projection( grid, 1, cx, cy);
        dg::x::HVec transferH( dg::evaluate(dg::zero, g3d_out));
        dg::x::DVec transferD( dg::evaluate(dg::zero, g3d_out));
        dg::x::HVec transferH2d = dg::evaluate( dg::zero, *g2d_out_ptr);
        dg::x::DVec transferD2d = dg::evaluate( dg::zero, *g2d_out_ptr);
        dg::x::HVec resultH = dg::evaluate( dg::zero, grid);
        dg::x::DVec resultD = dg::evaluate( dg::zero, grid);
        if( argc != 3 && argc != 4)
        {
            DG_RANK0 std::cerr << "ERROR: Wrong number of arguments for netcdf output!\nUsage: "
                    << argv[0]<<" [input.json] [output.nc]\n OR \n"
                    << argv[0]<<" [input.json] [output.nc] [initial.nc] "<<std::endl;
            dg::abort_program();
        }
        dg::file::NC_Error_Handle err;
        std::string file_name = argv[2];
        int ncid=-1;
        try{
            DG_RANK0 err = nc_create( file_name.data(), NC_NETCDF4|NC_CLOBBER, &ncid);
        }catch( std::exception& e)
        {
            DG_RANK0 std::cerr << "ERROR creating file "<<file_name<<std::endl;
            DG_RANK0 std::cerr << e.what()<<std::endl;
            dg::abort_program();
        }
        /// Set global attributes
        std::map<std::string, std::string> att;
        att["title"] = "Output file of feltor/src/feltor/feltor.cu";
        att["Conventions"] = "CF-1.8";
        ///Get local time and begin file history
        auto ttt = std::time(nullptr);
        std::ostringstream oss;
        ///time string  + program-name + args
        oss << std::put_time(std::localtime(&ttt), "%F %T %Z");
        for( int i=0; i<argc; i++) oss << " "<<argv[i];
        att["history"] = oss.str();
        att["comment"] = "Find more info in feltor/src/feltor/feltor.tex";
        att["source"] = "FELTOR";
        att["git-hash"] = GIT_HASH;
        att["git-branch"] = GIT_BRANCH;
        att["compile-time"] = COMPILE_TIME;
        att["references"] = "https://github.com/feltor-dev/feltor";
        att["inputfile"] = inputfile;
        for( auto pair : att)
            DG_RANK0 err = nc_put_att_text( ncid, NC_GLOBAL,
                pair.first.data(), pair.second.size(), pair.second.data());

        // Define dimensions (t,z,y,x)
        int dim_ids[4], restart_dim_ids[3], tvarID;
        DG_RANK0 err = dg::file::define_dimensions( ncid, &dim_ids[1], g3d_out,
                {"z", "y", "x"});
        if( !p.calibrate)
            DG_RANK0 err = dg::file::define_time( ncid, "time", dim_ids, &tvarID);
        DG_RANK0 err = dg::file::define_dimensions( ncid, restart_dim_ids, grid,
                {"zr", "yr", "xr"});
        int dim_ids3d[3] = {dim_ids[0], dim_ids[2], dim_ids[3]};
        bool write2d = true;
#ifdef WITH_MPI
        //only the globally first slice should write
        if( !(g3d_out.local().z0() - g3d_out.global().z0() < 1e-14) ) write2d = false;
#endif //WITH_MPI

        //create & output static 3d variables into file
        for ( auto& record : feltor::diagnostics3d_static_list)
        {
            int vecID;
            DG_RANK0 err = nc_def_var( ncid, record.name.data(), NC_DOUBLE, 3,
                &dim_ids[1], &vecID);
            DG_RANK0 err = nc_put_att_text( ncid, vecID,
                "long_name", record.long_name.size(), record.long_name.data());
            DG_RANK0 std::cout << "Computing "<<record.name<<"\n";
            record.function( transferH, var, g3d_out);
            //record.function( resultH, var, grid);
            //dg::blas2::symv( projectH, resultH, transferH);
            dg::file::put_var_double( ncid, vecID, g3d_out, transferH);
        }
        //create & output static 2d variables into file
        for ( auto& record : feltor::diagnostics2d_static_list)
        {
            int vecID;
            DG_RANK0 err = nc_def_var( ncid, record.name.data(), NC_DOUBLE, 2,
                &dim_ids[2], &vecID);
            DG_RANK0 err = nc_put_att_text( ncid, vecID,
                "long_name", record.long_name.size(), record.long_name.data());
            DG_RANK0 std::cout << "Computing2d "<<record.name<<"\n";
            //record.function( transferH, var, g3d_out); //ATTENTION: This does not work because feltor internal variables return full grid functions
            record.function( resultH, var, grid);
            dg::blas2::symv( projectH, resultH, transferH);
            if(write2d)dg::file::put_var_double( ncid, vecID, *g2d_out_ptr, transferH);
        }
        {
            // transition has to be done by hand
            int vecID;
            DG_RANK0 err = nc_def_var( ncid, "MagneticTransition", NC_DOUBLE, 2,
                &dim_ids[2], &vecID);
            std::string long_name = "The region where the magnetic field is modified";
            DG_RANK0 err = nc_put_att_text( ncid, vecID,
                "long_name", long_name.size(), long_name.data());
            DG_RANK0 std::cout << "Computing2d MagneticTransition\n";
            resultH = dg::pullback( transition, grid);
            dg::blas2::symv( projectH, resultH, transferH);
            if(write2d)dg::file::put_var_double( ncid, vecID, *g2d_out_ptr, transferH);
        }

        if( p.calibrate)
        {
            DG_RANK0 err = nc_close(ncid);
#ifdef WITH_MPI
            MPI_Finalize();
#endif //WITH_MPI
            return 0;
        }

        //Create field IDs
        // the vector ids
        std::map<std::string, int> id1d, id3d, id4d, restart_ids;
        for( auto& record : feltor::diagnostics3d_list)
        {
            std::string name = record.name;
            std::string long_name = record.long_name;
            id4d[name] = 0;//creates a new id4d entry for all processes
            DG_RANK0 err = nc_def_var( ncid, name.data(), NC_DOUBLE, 4, dim_ids,
                &id4d.at(name));
            DG_RANK0 err = nc_put_att_text( ncid, id4d.at(name), "long_name", long_name.size(),
                long_name.data());
        }
        for( auto& record : feltor::restart3d_list)
        {
            std::string name = record.name;
            std::string long_name = record.long_name;
            restart_ids[name] = 0;//creates a new entry for all processes
            DG_RANK0 err = nc_def_var( ncid, name.data(), NC_DOUBLE, 3,
                    restart_dim_ids, &restart_ids.at(name));
            DG_RANK0 err = nc_put_att_text( ncid, restart_ids.at(name),
                    "long_name", long_name.size(), long_name.data());
        }
        for( auto& record : feltor::diagnostics2d_list)
        {
            std::string name = record.name + "_ta2d";
            std::string long_name = record.long_name + " (Toroidal average)";
            id3d[name] = 0;//creates a new id3d entry for all processes
            DG_RANK0 err = nc_def_var( ncid, name.data(), NC_DOUBLE, 3, dim_ids3d,
                &id3d.at(name));
            DG_RANK0 err = nc_put_att_text( ncid, id3d.at(name), "long_name",
                    long_name.size(), long_name.data());

            name = record.name + "_2d";
            long_name = record.long_name + " (Evaluated on phi = 0 plane)";
            id3d[name] = 0;
            DG_RANK0 err = nc_def_var( ncid, name.data(), NC_DOUBLE, 3,
                    dim_ids3d, &id3d.at(name));
            DG_RANK0 err = nc_put_att_text( ncid, id3d.at(name), "long_name",
                    long_name.size(), long_name.data());
        }
        for( auto& record : feltor::diagnostics1d_list)
        {
            std::string name = record.name;
            std::string long_name = record.long_name;
            id1d[name] = 0;
            DG_RANK0 err = nc_def_var( ncid, name.data(), NC_DOUBLE, 1,
                    &dim_ids[0], &id1d.at(name));
            DG_RANK0 err = nc_put_att_text( ncid, id1d.at(name), "long_name",
                    long_name.size(), long_name.data());
        }
        ///////////////////////////////////first output/////////////////////////
        DG_RANK0 std::cout << "First output ... \n";
        //first, update feltor (to get potential etc.)
        {
            std::array<std::array<dg::x::DVec,2>,2> y1(y0);
            try{
                feltor( time, y0, y1);
            } catch( dg::Fail& fail) {
                DG_RANK0 std::cerr << "CG failed to converge in first step to "
                                  <<fail.epsilon()<<std::endl;
                DG_RANK0 err = nc_close(ncid);
                dg::abort_program();
            }
        }

        size_t start = 0, count = 1;
        DG_RANK0 err = nc_put_vara_double( ncid, tvarID, &start, &count, &time);
        for( auto& record : feltor::diagnostics3d_list)
        {
            record.function( resultD, var);
            dg::blas2::symv( projectD, resultD, transferD);
            dg::assign( transferD, transferH);
            dg::file::put_vara_double( ncid, id4d.at(record.name), start, g3d_out, transferH);
        }
        for( auto& record : feltor::restart3d_list)
        {
            record.function( resultD, var);
            dg::assign( resultD, resultH);
            dg::file::put_var_double( ncid, restart_ids.at(record.name), grid, resultH);
        }
        for( auto& record : feltor::diagnostics2d_list)
        {
            dg::Timer tti;
            tti.tic();
            record.function( resultD, var);
            dg::blas2::symv( projectD, resultD, transferD);

            //toroidal average
            std::string name = record.name + "_ta2d";
            dg::assign( transferD, transferH);
            toroidal_average( transferH, transferH2d, false);
            //create and init Simpsons for time integrals
            if( record.integral) time_integrals[name].init( time, transferH2d);
            tti.toc();
            DG_RANK0 std::cout<< name << " Computing average took "<<tti.diff()<<"\n";
            tti.tic();
            if(write2d) dg::file::put_vara_double( ncid, id3d.at(name), start, *g2d_out_ptr, transferH2d);
            tti.toc();
            DG_RANK0 std::cout<< name << " 2d output took "<<tti.diff()<<"\n";
            tti.tic();

            // and a slice
            name = record.name + "_2d";
            feltor::slice_vector3d( transferD, transferD2d, local_size2d);
            dg::assign( transferD2d, transferH2d);
            if( record.integral) time_integrals[name].init( time, transferH2d);
            if(write2d) dg::file::put_vara_double( ncid, id3d.at(name), start, *g2d_out_ptr, transferH2d);
            tti.toc();
            DG_RANK0 std::cout<< name << " 2d output took "<<tti.diff()<<"\n";
        }
        for( auto& record : feltor::diagnostics1d_list)
        {
            double result = record.function( var);
            DG_RANK0 nc_put_vara_double( ncid, id1d.at(record.name), &start, &count, &result);
        }
        DG_RANK0 err = nc_close(ncid);
        DG_RANK0 std::cout << "First write successful!\n";
        ///////////////////////////////Timeloop/////////////////////////////////

        t.tic();
        unsigned step = 0;
        unsigned maxout = js["output"].get( "maxout", 0).asUInt();
        double Tend = 0, deltaT = 0, t_output = 0;
        std::string output_mode = "free";
        if( p.timestepper == "multistep")
            multistep.init( feltor, time, y0, dt);
        if( p.timestepper == "multistep-imex")
            multistep_imex.init( std::tie( feltor, implicit, solver), time, y0,
                    dt);
        else if( p.timestepper == "adaptive" || p.timestepper == "adaptive-imex")
        {
            output_mode = js["timestepper"].get(
                    "output-mode", "equidistant").asString();
            if( output_mode == "equidistant")
            {
                Tend = js["timestepper"].get( "Tend", 1).asDouble();
                deltaT = Tend/(double)(maxout*p.itstp);
                t_output = time + deltaT;
            }
            else if( !(output_mode == "free"))
                throw std::runtime_error( "timestepper: output-mode "+output_mode+" not recognized!\n");

        }
        bool abort = false;
        for( unsigned i=1; i<=maxout; i++)
        {
            dg::Timer ti;
            ti.tic();
            for( unsigned j=0; j<p.itstp; j++)
            {
                try{
                if( output_mode == "equidistant")
                {
                    while( time < t_output)
                    {
                        if( time + dt > t_output)
                            dt = t_output-time;
                        if( p.timestepper == "adaptive")
                        {
                            do{
                                adapt.step( feltor, time, y0, time, y0, dt,
                                        dg::pid_control, dg::l2norm, rtol,
                                        atol, reject_limit);
                                if( adapt.failed())
                                    var.nfailed++;
                                if( dt < 1e-6)
                                    throw dg::Error(dg::Message(_ping_)<<"Adaptive failed to converge! dt = "<<std::scientific<<dt);
                            }while( adapt.failed());
                        }
                        else if ( p.timestepper == "adaptive-imex")
                            do{
                                adapt_ark.step( std::tie( feltor, implicit,
                                            solver), time, y0, time, y0, dt,
                                        dg::pid_control, dg::l2norm, rtol,
                                        atol, reject_limit);
                                if( adapt_ark.failed())
                                    var.nfailed++;
                                if( dt < 1e-6)
                                    throw dg::Error(dg::Message(_ping_)<<"Adaptive failed to converge! dt = "<<std::scientific<<dt);
                            }while( adapt_ark.failed());
                        DG_RANK0 std::cout << "## time "<<time<<" dt "<<dt<<" t_out "<<t_output<<" step "<<step<<" failed "<<var.nfailed<<"\n";
                        step++;
                    }
                    t_output += deltaT;
                }
                else
                    for( unsigned k=0; k<p.inner_loop; k++)
                    {
                        if( p.timestepper == "adaptive")
                        {
                            do{
                                adapt.step( feltor, time, y0, time, y0, dt,
                                        dg::pid_control, dg::l2norm, rtol,
                                        atol, reject_limit);
                                if( adapt.failed())
                                    var.nfailed++;
                                if( dt < 1e-6)
                                    throw dg::Error(dg::Message(_ping_)<<"Adaptive failed to converge! dt = "<<std::scientific<<dt);
                            }while( adapt.failed());
                        }
                        else if( p.timestepper == "adaptive-imex")
                            do{
                                adapt_ark.step( std::tie( feltor, implicit,
                                            solver), time, y0, time, y0, dt,
                                        dg::pid_control, dg::l2norm, rtol,
                                        atol, reject_limit);
                                if( adapt_ark.failed())
                                    var.nfailed++;
                                if( dt < 1e-6)
                                    throw dg::Error(dg::Message(_ping_)<<"Adaptive failed to converge! dt = "<<std::scientific<<dt);
                            }while( adapt_ark.failed());
                        else if ( p.timestepper == "multistep")
                            multistep.step( feltor, time, y0);
                        else if ( p.timestepper == "multistep-imex")
                            multistep_imex.step( std::tie( feltor, implicit,
                                        solver), time, y0);
                        step++;
                    }
                }
                catch( dg::Fail& fail){ // a specific exception
                    DG_RANK0 std::cerr << "ERROR failed to converge to "<<fail.epsilon()<<"\n";
                    DG_RANK0 std::cerr << "Does simulation respect CFL condition?"<<std::endl;
                    DG_RANK0 std::cerr << "Writing last Output and exit..."<<std::endl;
                    abort = true; // exit gracefully
                }
                catch( std::exception& fail) { // more generic exception
                    DG_RANK0 std::cerr << "ERROR in timestepper\n";
                    DG_RANK0 std::cerr << fail.what()<<std::endl;
                    DG_RANK0 std::cerr << "Writing last Output and exit..."<<std::endl;
                    abort = true;
                }
                dg::Timer tti;
                tti.tic();
                for( auto& record : feltor::diagnostics2d_list)
                {
                    if( record.integral)
                    {
                        record.function( resultD, var);
                        dg::blas2::symv( projectD, resultD, transferD);
                        //toroidal average and add to time integral
                        dg::assign( transferD, transferH);
                        toroidal_average( transferH, transferH2d, false);
                        time_integrals.at(record.name+"_ta2d").add( time,
                                transferH2d);

                        // 2d data of plane varphi = 0
                        feltor::slice_vector3d( transferD, transferD2d,
                                local_size2d);
                        dg::assign( transferD2d, transferH2d);
                        time_integrals.at(record.name+"_2d").add( time,
                                transferH2d);
                    }
                }

                DG_RANK0 std::cout << "\tTime "<<time<<"\n";
                double max_ue = dg::blas1::reduce(
                    feltor.velocity(0), 0., dg::AbsMax<double>() );
                DG_RANK0 std::cout << "\tMaximum ue "<<max_ue<<"\n";
                if( p.timestepper == "adaptive" || p.timestepper == "adaptive-imex")
                {
                    DG_RANK0 std::cout << "\tdt "<<dt<<"\n";
                    DG_RANK0 std::cout << "\tfailed "<<var.nfailed<<"\n";
                }

                tti.toc();
                DG_RANK0 std::cout << " Time for internal diagnostics "<<tti.diff()<<"s\n";
                if( abort) break;
            }
            ti.toc();
            var.duration = ti.diff();
            // Does not work due to direct application of Laplace
            // The Laplacian of Aparallel looks smooth in paraview
            ////----------------Test if ampere equation holds
            //if( p.beta != 0 && !abort)
            //{
            //    feltor.compute_lapMperpA( resultD);
            //    double norm  = dg::blas2::dot( resultD, feltor.vol3d(), resultD);
            //    dg::blas1::pointwiseDot( -p.beta,
            //        feltor.density(0), feltor.velocity(0), p.beta,
            //        feltor.density(1), feltor.velocity(1), -1., resultD);
            //    double error = dg::blas2::dot( resultD, feltor.vol3d(), resultD);
            //    DG_RANK0 std::cout << "\tRel. Error Ampere "<<sqrt(error/norm) <<"\n";
            //}
            if( output_mode == "free")
            {
                DG_RANK0 std::cout << "\n\t Step "<<step <<" of "
                            << p.inner_loop*p.itstp*maxout << " at time "<<time;
                DG_RANK0 std::cout << "\n\t Average time for one step: "
                            << var.duration/(double)p.itstp/(double)p.inner_loop<<"s";
            }
            else
                DG_RANK0 std::cout << "\n\t Step: Time "<<time <<" of " << Tend;
                DG_RANK0 std::cout << "\n\t Average time for one inner loop: "
                            << var.duration/(double)p.itstp<<"s";

            ti.tic();
            //////////////////////////write fields////////////////////////
            start = i;
            DG_RANK0 err = nc_open(file_name.data(), NC_WRITE, &ncid);
            DG_RANK0 err = nc_put_vara_double( ncid, tvarID, &start, &count, &time);
            for( auto& record : feltor::diagnostics3d_list)
            {
                record.function( resultD, var);
                dg::blas2::symv( projectD, resultD, transferD);
                dg::assign( transferD, transferH);
                dg::file::put_vara_double( ncid, id4d.at(record.name), start,
                        g3d_out, transferH);
            }
            for( auto& record : feltor::restart3d_list)
            {
                record.function( resultD, var);
                dg::assign( resultD, resultH);
                dg::file::put_var_double( ncid, restart_ids.at(record.name),
                        grid, resultH);
            }
            for( auto& record : feltor::diagnostics2d_list)
            {
                if(record.integral) // we already computed the output...
                {
                    std::string name = record.name+"_ta2d";
                    transferH2d = time_integrals.at(name).get_integral();
                    time_integrals.at(name).flush();
                    if(write2d) dg::file::put_vara_double( ncid, id3d.at(name),
                            start, *g2d_out_ptr, transferH2d);

                    name = record.name+"_2d";
                    transferH2d = time_integrals.at(name).get_integral( );
                    time_integrals.at(name).flush( );
                    if(write2d) dg::file::put_vara_double( ncid, id3d.at(name),
                            start, *g2d_out_ptr, transferH2d);
                }
                else // compute from scratch
                {
                    record.function( resultD, var);
                    dg::blas2::symv( projectD, resultD, transferD);

                    std::string name = record.name+"_ta2d";
                    dg::assign( transferD, transferH);
                    toroidal_average( transferH, transferH2d, false);
                    if(write2d) dg::file::put_vara_double( ncid, id3d.at(name),
                            start, *g2d_out_ptr, transferH2d);

                    // 2d data of plane varphi = 0
                    name = record.name+"_2d";
                    feltor::slice_vector3d( transferD, transferD2d, local_size2d);
                    dg::assign( transferD2d, transferH2d);
                    if(write2d) dg::file::put_vara_double( ncid, id3d.at(name),
                            start, *g2d_out_ptr, transferH2d);
                }
            }
            for( auto& record : feltor::diagnostics1d_list)
            {
                double result = record.function( var);
                DG_RANK0 nc_put_vara_double( ncid, id1d.at(record.name), &start, &count, &result);
            }
            DG_RANK0 err = nc_close(ncid);
            ti.toc();
            DG_RANK0 std::cout << "\n\t Time for output: "<<ti.diff()<<"s\n\n"<<std::flush;
            if( abort) break;
        }
        t.toc();
        unsigned hour = (unsigned)floor(t.diff()/3600);
        unsigned minute = (unsigned)floor( (t.diff() - hour*3600)/60);
        double second = t.diff() - hour*3600 - minute*60;
        DG_RANK0 std::cout << std::fixed << std::setprecision(2) <<std::setfill('0');
        DG_RANK0 std::cout <<"Computation Time \t"<<hour<<":"<<std::setw(2)<<minute<<":"<<second<<"\n";
        DG_RANK0 std::cout <<"which is         \t"<<t.diff()/p.itstp/maxout<<"s/inner loop\n";
    }
#ifndef WITHOUT_GLFW
    if( p.output == "glfw")
    {
        dg::Timer t;
        unsigned step = 0;

        std::map<std::string, const dg::x::DVec* > v4d;
        v4d["ne-1 / "] = &feltor.density(0),  v4d["ni-1 / "] = &feltor.density(1);
        v4d["Ue / "]   = &feltor.velocity(0), v4d["Ui / "]   = &feltor.velocity(1);
        v4d["Phi / "] = &feltor.potential(0); v4d["Apar / "] = &feltor.aparallel();
        /////////////////////////set up transfer for glfw
        dg::DVec dvisual( grid.size(), 0.);
        dg::HVec hvisual( grid.size(), 0.), visual(hvisual), avisual(hvisual);
        dg::IHMatrix equi = dg::create::backscatter( grid);
        draw::ColorMapRedBlueExtMinMax colors(-1.0, 1.0);

        /////////glfw initialisation ////////////////////////////////////////////
        //
        std::stringstream title;
        unsigned red = js["output"]["window"].get("reduction", 1).asUInt();
        double rows = js["output"]["window"]["rows"].asDouble(), cols = p.Nz/red+1,
               width = js["output"]["window"]["width"].asDouble(), height = js["output"]["window"]["height"].asDouble();
        if ( p.symmetric ) cols = rows, rows = 1;
        GLFWwindow* w = draw::glfwInitAndCreateWindow( cols*width, rows*height, "");
        draw::RenderHostData render(rows, cols);

        std::cout << "Begin computation \n";
        std::cout << std::scientific << std::setprecision( 2);
        dg::Average<dg::HVec> toroidal_average( grid, dg::coo3d::z, "simple");
        title << std::setprecision(2) << std::scientific;
        while ( !glfwWindowShouldClose( w ))
        {
            title << std::fixed;
            title << "t = "<<time<<"   ";
            for( auto pair : v4d)
            {
                if(pair.first == "Phi / ")
                {
                    //dg::assign( feltor.lapMperpP(0), hvisual);
                    dg::assign( *pair.second, hvisual);
                }
                else if(pair.first == "ne-1 / " || pair.first == "ni-1 / ")
                {
                    dg::assign( *pair.second, hvisual);
                    //dg::blas1::axpby( 1., hvisual, -1., profile, hvisual);
                }
                else
                    dg::assign( *pair.second, hvisual);
                dg::blas2::gemv( equi, hvisual, visual);
                colors.scalemax() = dg::blas1::reduce(
                    visual, 0., dg::AbsMax<double>() );
                colors.scalemin() = -colors.scalemax();
                title <<pair.first << colors.scalemax()<<"   ";
                if ( p.symmetric )
                    render.renderQuad( hvisual, grid.n()*grid.Nx(),
                                                grid.n()*grid.Ny(), colors);
                else
                {
                    for( unsigned k=0; k<p.Nz/red;k++)
                    {
                        unsigned size=grid.n()*grid.n()*grid.Nx()*grid.Ny();
                        dg::HVec part( visual.begin() +  k*red   *size,
                                       visual.begin() + (k*red+1)*size);
                        render.renderQuad( part, grid.n()*grid.Nx(),
                                                 grid.n()*grid.Ny(), colors);
                    }
                    dg::blas1::scal(avisual,0.);
                    toroidal_average(visual,avisual);
                    render.renderQuad( avisual, grid.n()*grid.Nx(),
                                                grid.n()*grid.Ny(), colors);
                }
            }
            glfwSetWindowTitle(w,title.str().c_str());
            title.str("");
            glfwPollEvents();
            glfwSwapBuffers( w);

            //step
            t.tic();
            if( p.timestepper == "multistep")
                multistep.init( feltor, time, y0, dt);
            if( p.timestepper == "multistep-imex")
                multistep_imex.init( std::tie( feltor, implicit, solver), time,
                        y0, dt);
            for( unsigned i=0; i<p.itstp; i++)
            {
                for( unsigned k=0; k<p.inner_loop; k++)
                {
                    try{
                        if( p.timestepper == "adaptive")
                        {
                            do{
                                adapt.step( feltor, time, y0, time, y0, dt,
                                        dg::pid_control, dg::l2norm, rtol,
                                        atol);
                                if( adapt.failed())
                                    var.nfailed++;
                                if( dt < 1e-6)
                                    throw dg::Error(dg::Message(_ping_)<<"Adaptive failed to converge! dt = "<<std::scientific<<dt);
                            }while( adapt.failed());
                        }
                        else if( p.timestepper == "adaptive-imex")
                            do{
                                adapt_ark.step( std::tie( feltor, implicit,
                                            solver), time, y0, time, y0, dt,
                                        dg::pid_control, dg::l2norm, rtol,
                                        atol);
                                if( adapt_ark.failed())
                                    var.nfailed++;
                                if( dt < 1e-6)
                                    throw dg::Error(dg::Message(_ping_)<<"Adaptive failed to converge! dt = "<<std::scientific<<dt);
                            }while( adapt_ark.failed());
                        else if ( p.timestepper == "multistep")
                            multistep.step( feltor, time, y0);
                        else if ( p.timestepper == "multistep-imex")
                            multistep_imex.step( std::tie( feltor, implicit,
                                        solver), time, y0);
                    }
                    catch( dg::Fail& fail) {
                        std::cerr << "CG failed to converge to "<<fail.epsilon()<<"\n";
                        std::cerr << "Does Simulation respect CFL condition?\n";
                        glfwSetWindowShouldClose( w, GL_TRUE);
                        break;
                    }
                    step++;
                }

                std::cout << "\tTime "<<time<<"\n";
                double max_ue = dg::blas1::reduce(
                    feltor.velocity(0), 0., dg::AbsMax<double>() );
                std::cout << "\tMaximum ue "<<max_ue<<"\n";
                if( p.timestepper == "adaptive" || p.timestepper == "adaptive-imex")
                {
                    std::cout << "\tdt "<<dt<<"\n";
                    std::cout << "\tfailed "<<var.nfailed<<"\n";
                }
                //----------------Test if ampere equation holds
                // Does not work due to direct application of Laplace
                // The Laplacian of Aparallel looks smooth in paraview
                //if( p.beta != 0)
                //{
                //    feltor.compute_lapMperpA( dvisual);
                //    double norm  = dg::blas2::dot( dvisual, feltor.vol3d(), dvisual);
                //    dg::blas1::pointwiseDot( -p.beta,
                //        feltor.density(0), feltor.velocity(0), p.beta,
                //        feltor.density(1), feltor.velocity(1), -1., dvisual);
                //    double error = dg::blas2::dot( dvisual, feltor.vol3d(), dvisual);
                //    DG_RANK0 std::cout << "\tRel. Error Ampere "<<sqrt(error/norm) <<"\n";
                //}

            }
            t.toc();
            std::cout << "\n\t Step "<<step << " at time  "<<time;
            std::cout << "\n\t Average time for one step: "<<t.diff()/(double)p.itstp/(double)p.inner_loop<<"\n\n";
        }
        glfwTerminate();
    }
#endif //WITHOUT_GLFW
#ifdef WITH_MPI
    MPI_Finalize();
#endif //WITH_MPI

    return 0;

}
