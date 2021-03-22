#include <iostream>
#include <iomanip>
#include <vector>
#include <sstream>
#include <cmath>

#ifdef WITH_MPI
#include <mpi.h>
#endif //WITH_MPI

#ifndef WITHOUT_GLFW
#include "draw/host_window.h"
#endif // WITHOUT_GLFW

#include "dg/file/file.h"

#include "reconnection.h"
#include "init.h"
#include "diag.h"

int main( int argc, char* argv[])
{
#ifdef WITH_MPI
    ////////////////////////////////setup MPI///////////////////////////////
    dg::mpi_init( argc, argv);
    MPI_Comm comm;
    dg::mpi_init2d( dg::DIR, dg::PER, comm, std::cin, true);
    int rank;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank);
#endif //WITH_MPI

    ////Parameter initialisation ////////////////////////////////////////////
    Json::Value js;
    if( argc == 1)
        dg::file::file2Json( "input/default.json", js, dg::file::comments::are_discarded);
    else
        dg::file::file2Json( argv[1], js);
    DG_RANK0 std::cout << js <<std::endl;

    const asela::Parameters p( js);
    dg::file::WrappedJsonValue ws ( js, dg::file::error::is_throw);

    //////////////////////////////////////////////////////////////////////////
    //Make grid

    dg::x::CartesianGrid2d grid( -p.lxhalf, p.lxhalf, -p.lyhalf, p.lyhalf , p.n, p.Nx, p.Ny, dg::DIR, dg::PER
        #ifdef WITH_MPI
        , comm
        #endif //WITH_MPI
        );
    DG_RANK0 std::cout << "Constructing Explicit...\n";
    asela::Asela<dg::x::CartesianGrid2d, dg::x::DMatrix, dg::x::DVec> asela( grid, p);
    DG_RANK0 std::cout << "Done!\n";

    /// //////////////////The initial field///////////////////////////////////////////
    double time = 0.;
    std::array<std::array<dg::x::DVec,2>,2> y0;
    try{
        y0 = asela::initial_conditions(asela, grid, p, ws );
    }catch ( std::exception& error){
        DG_RANK0 std::cerr << "Error in input file\n ";
        DG_RANK0 std::cerr << error.what();
#ifdef WITH_MPI
        MPI_Abort(MPI_COMM_WORLD, -1);
#endif //WITH_MPI
        return -1;
    }
    DG_RANK0 std::cout << "Initialize time stepper..." << std::endl;
    dg::ExplicitMultistep< std::array<std::array<dg::x::DVec,2>,2>> multistep;
    dg::Adaptive< dg::ERKStep< std::array<std::array<dg::x::DVec,2>,2>>> adapt;
    double rtol = 0., atol = 0., dt = 0.;

    unsigned step = 0;
    if( p.timestepper == "multistep")
    {
        std::string tableau = ws[ "timestepper"]["tableau"].asString("TVB-3-3");
        multistep.construct( tableau, y0);
        dt = ws[ "timestepper"]["dt"].asDouble( 20);
        multistep.init( asela, time, y0, dt);
    }
    else if (p.timestepper == "adaptive")
    {
        std::string tableau = ws[ "timestepper"]["tableau"].asString( "Tsitouras09-7-4-5");
        adapt.construct( tableau, y0);
        rtol = ws[ "timestepper"][ "rtol"].asDouble( 1e-7);
        atol = ws[ "timestepper"][ "atol"].asDouble( 1e-10);
        dt = 1e-3; //that should be a small enough initial guess
    }
    else
    {
        DG_RANK0 std::cerr<<"Error: Unrecognized timestepper: '"<<p.timestepper<<"'! Exit now!";
#ifdef WITH_MPI
        MPI_Abort(MPI_COMM_WORLD, -1);
#endif //WITH_MPI
        return -1;
    }



    DG_RANK0 std::cout << "Done!\n";


    /// ////////////Init diagnostics ////////////////////
    asela::Variables var = {asela, p, y0[0]};
    dg::Timer t;
    t.tic();
    {
        std::array<std::array<dg::x::DVec,2>,2> y1 = y0;
        asela( 0., y0, y1);
    }
    t.toc();
    var.duration = t.diff();
    t.tic();

    DG_RANK0 std::cout << "Begin computation \n";
    DG_RANK0 std::cout << std::scientific << std::setprecision( 2);
    unsigned maxout = ws["output"]["maxout"].asUInt(100);
    unsigned itstp = ws["output"]["itstp"].asUInt(5);
    std::string output = ws[ "output"]["type"].asString("glfw");
#ifndef WITHOUT_GLFW
    if( "glfw" == output)
    {
        /////////glfw initialisation ////////////////////////////////////////////
        dg::file::file2Json( "window_params.json", js, dg::file::comments::are_discarded);
        GLFWwindow* w = draw::glfwInitAndCreateWindow( js["width"].asDouble(), js["height"].asDouble(), "");
        draw::RenderHostData render(js["rows"].asDouble(), js["cols"].asDouble());
        //create visualisation vectors
        dg::DVec visual( grid.size()), temp(visual);
        dg::HVec hvisual( grid.size());
        //transform vector to an equidistant grid
        std::stringstream title;
        draw::ColorMapRedBlueExtMinMax colors( -1.0, 1.0);
        dg::IDMatrix equidistant = dg::create::backscatter( grid );
        // the things to plot:
        std::map<std::string, const dg::DVec* > v2d;
        v2d["ne-1 / "] = &y0[0][0],  v2d["ni-1 / "] = &y0[0][1];
        v2d["Ue / "]   = &asela.velocity(0), v2d["Ui / "]   = &asela.velocity(1);
        v2d["Phi / "] = &asela.potential(0); v2d["Apar / "] = &asela.aparallel(0);
        v2d["Vor / "] = &asela.potential(0); v2d["j / "]    = &asela.aparallel(0);

        while ( !glfwWindowShouldClose( w ))
        {
            for( auto pair : v2d)
            {
                if( pair.first == "Vor / " || pair.first == "j / ")
                {
                    asela.compute_lapM( 1., *pair.second, 0., temp);
                    if( pair.first == "j / ")
                        dg::blas1::scal( temp, 1./p.beta);
                    dg::blas2::gemv( equidistant, temp, visual);
                }
                else
                    dg::blas2::gemv( equidistant, *pair.second, visual);
                dg::assign( visual, hvisual);
                colors.scalemax() = dg::blas1::reduce(
                    hvisual, 0., dg::AbsMax<double>() );
                colors.scalemin() = -colors.scalemax();
                title << std::setprecision(2) << std::scientific;
                title <<pair.first << colors.scalemax()<<"   ";
                render.renderQuad( hvisual, grid.n()*grid.Nx(),
                        grid.n()*grid.Ny(), colors);
            }
            title << std::fixed;
            title << " &&   time = "<<time;
            glfwSetWindowTitle(w,title.str().c_str());
            title.str("");
            glfwPollEvents();
            glfwSwapBuffers( w);

            //step
            dg::Timer ti;
            ti.tic();
            for( unsigned i=0; i<itstp; i++)
            {
                try{
                    if( p.timestepper == "adaptive")
                        adapt.step( asela, time, y0, time, y0, dt, dg::pid_control, dg::l2norm, rtol, atol);
                    if( p.timestepper == "multistep")
                        multistep.step( asela, time, y0);
                }
                catch( dg::Fail& fail) {
                    std::cerr << "CG failed to converge to "<<fail.epsilon()<<"\n";
                    std::cerr << "Does Simulation respect CFL condition?\n";
                    glfwSetWindowShouldClose( w, GL_TRUE);
                    break;
                }
                step++;
            }
            ti.toc();
            std::cout << "\n\t Step "<<step;
            std::cout << "\n\t Average time for one step: "<<ti.diff()/(double)itstp<<"s\n\n";
        }
        glfwTerminate();
    }
#endif //WITHOUT_GLFW
    if( "netcdf" == output)
    {
        std::string inputfile = js.toStyledString(); //save input without comments, which is important if netcdf file is later read by another parser
        std::string outputfile;
        if( argc == 1 || argc == 2)
            outputfile = "reconnection.nc";
        else
            outputfile = argv[2];
        /// //////////////////////set up netcdf/////////////////////////////////////
        dg::file::NC_Error_Handle err;
        int ncid=-1;
        try{
            err = nc_create( outputfile.c_str(),NC_NETCDF4|NC_CLOBBER, &ncid);
        }catch( std::exception& e)
        {
            std::cerr << "ERROR creating file "<<outputfile<<std::endl;
            std::cerr << e.what()<<std::endl;
           return -1;
        }
        /// Set global attributes
        std::map<std::string, std::string> att;
        att["title"] = "Output file of feltor/src/reco2D/reconnection.cu";
        att["Conventions"] = "CF-1.7";
        ///Get local time and begin file history
        auto ttt = std::time(nullptr);
        auto tm = *std::localtime(&ttt);

        std::ostringstream oss;
        ///time string  + program-name + args
        oss << std::put_time(&tm, "%Y-%m-%d %H:%M:%S");
        for( int i=0; i<argc; i++) oss << " "<<argv[i];
        att["history"] = oss.str();
        att["comment"] = "Find more info in feltor/src/reco2D/reconnection.tex";
        att["source"] = "FELTOR";
        att["references"] = "https://github.com/feltor-dev/feltor";
        att["inputfile"] = inputfile;
        for( auto pair : att)
            DG_RANK0 err = nc_put_att_text( ncid, NC_GLOBAL,
                pair.first.data(), pair.second.size(), pair.second.data());

        int dim_ids[3], tvarID;
        std::map<std::string, int> id1d, id3d;
        unsigned n_out     = ws[ "output"]["n"].asUInt( 3);
        unsigned Nx_out    = ws[ "output"]["Nx"].asUInt( 48);
        unsigned Ny_out    = ws[ "output"]["Ny"].asUInt( 48);
        dg::x::CartesianGrid2d grid_out( -p.lxhalf, p.lxhalf, -p.lyhalf, p.lyhalf, n_out, Nx_out, Ny_out, dg::DIR, dg::PER
            #ifdef WITH_MPI
            , comm
            #endif //WITH_MPI
            );
        dg::x::IHMatrix projection = dg::create::interpolation( grid_out, grid);
        err = dg::file::define_dimensions( ncid, dim_ids, &tvarID, grid_out,
                {"time", "y", "x"});

        //Create field IDs
        for( auto& record : asela::diagnostics2d_list)
        {
            std::string name = record.name;
            std::string long_name = record.long_name;
            id3d[name] = 0;
            DG_RANK0 err = nc_def_var( ncid, name.data(), NC_DOUBLE, 3, dim_ids,
                    &id3d.at(name));
            DG_RANK0 err = nc_put_att_text( ncid, id3d.at(name), "long_name",
                    long_name.size(), long_name.data());
        }
        for( auto& record : asela::diagnostics2d_list)
        {
            std::string name = record.name + "_1d";
            std::string long_name = record.long_name + " (Volume integrated)";
            id1d[name] = 0;
            DG_RANK0 err = nc_def_var( ncid, name.data(), NC_DOUBLE, 1, &dim_ids[0],
                &id1d.at(name));
            DG_RANK0 err = nc_put_att_text( ncid, id1d.at(name), "long_name",
                    long_name.size(), long_name.data());
        }
        for( auto& record : asela::diagnostics1d_list)
        {
            std::string name = record.name;
            std::string long_name = record.long_name;
            id1d[name] = 0;
            DG_RANK0 err = nc_def_var( ncid, name.data(), NC_DOUBLE, 1, &dim_ids[0],
                &id1d.at(name));
            DG_RANK0 err = nc_put_att_text( ncid, id1d.at(name), "long_name", long_name.size(),
                long_name.data());
        }
        dg::x::DVec volume = dg::create::volume( grid);
        dg::x::DVec resultD = volume;
        dg::x::HVec resultH = dg::evaluate( dg::zero, grid);
        dg::x::HVec transferH = dg::evaluate( dg::zero, grid_out);
        for( auto& record : asela::diagnostics2d_static_list)
        {
            std::string name = record.name;
            std::string long_name = record.long_name;
            int staticID = 0;
            DG_RANK0 err = nc_def_var( ncid, name.data(), NC_DOUBLE, 2, &dim_ids[1],
                &staticID);
            DG_RANK0 err = nc_put_att_text( ncid, staticID, "long_name", long_name.size(),
                long_name.data());
            DG_RANK0 err = nc_enddef(ncid);
            record.function( resultD, var);
            dg::assign( resultD, resultH);
            dg::blas2::gemv( projection, resultH, transferH);
            dg::file::put_var_double( ncid, staticID, grid_out, transferH);
            DG_RANK0 err = nc_redef(ncid);
        }
        DG_RANK0 err = nc_enddef(ncid);
        size_t start = {0};
        size_t count = {1};
        ///////////////////////////////////first output/////////////////////////
        for( auto& record : asela::diagnostics2d_list)
        {
            record.function( resultD, var);
            double result = dg::blas1::dot( volume, resultD);
            dg::assign( resultD, resultH);
            dg::blas2::gemv( projection, resultH, transferH);
            dg::file::put_vara_double( ncid, id3d.at(record.name), start,
                    grid_out, transferH);
            DG_RANK0 err = nc_put_vara_double( ncid, id1d.at(record.name+"_1d"),
                    &start, &count, &result);
        }
        for( auto& record : asela::diagnostics1d_list)
        {
            double result = record.function( var);
            DG_RANK0 err = nc_put_vara_double( ncid, id1d.at(record.name), &start, &count, &result);
        }
        DG_RANK0 err = nc_put_vara_double( ncid, tvarID, &start, &count, &time);
        DG_RANK0 err = nc_close( ncid);
        ///////////////////////////////////timeloop/////////////////////////
        for( unsigned i=1; i<=maxout; i++)
        {
            dg::Timer ti;
            ti.tic();
            for( unsigned j=0; j<itstp; j++)
            {
                try{
                    if( p.timestepper == "adaptive")
                        adapt.step( asela, time, y0, time, y0, dt, dg::pid_control, dg::l2norm, rtol, atol);
                    if( p.timestepper == "multistep")
                        multistep.step( asela, time, y0);
                }
                catch( dg::Fail& fail) {
                    DG_RANK0 std::cerr << "ERROR failed to converge to "<<fail.epsilon()<<"\n";
                    DG_RANK0 std::cerr << "Does simulation respect CFL condition?"<<std::endl;
#ifdef WITH_MPI
                    MPI_Abort(MPI_COMM_WORLD, -1);
#endif //WITH_MPI
                    return -1;
                }
            }
            ti.toc();
            var.duration = ti.diff() / (double) itstp;
            step+=itstp;
            DG_RANK0 std::cout << "\n\t Step "<<step <<" of "<<itstp*maxout <<" at time "<<time << " with current timestep "<<dt;
            DG_RANK0 std::cout << "\n\t Average time for one step: "<<ti.diff()/(double)itstp<<"s\n\n"<<std::flush;
            //output all fields
            ti.tic();
            start = i;
            DG_RANK0 err = nc_open(outputfile.data(), NC_WRITE, &ncid);
            DG_RANK0 err = nc_put_vara_double( ncid, tvarID, &start, &count, &time);
            for( auto& record : asela::diagnostics2d_list)
            {
                record.function( resultD, var);
                double result = dg::blas1::dot( volume, resultD);
                dg::assign( resultD, resultH);
                dg::blas2::gemv( projection, resultH, transferH);
                dg::file::put_vara_double( ncid, id3d.at(record.name), start, grid_out, transferH);
                DG_RANK0 err = nc_put_vara_double( ncid, id1d.at(record.name+"_1d"), &start, &count, &result);
            }
            for( auto& record : asela::diagnostics1d_list)
            {
                double result = record.function( var);
                DG_RANK0 err = nc_put_vara_double( ncid, id1d.at(record.name), &start, &count, &result);
            }
            DG_RANK0 err = nc_close( ncid);
            ti.toc();
            DG_RANK0 std::cout << "\n\t Time for output: "<<ti.diff()<<"s\n\n"<<std::flush;
        }
    }
    if( !("netcdf" == output) && !("glfw" == output))
    {
        DG_RANK0 std::cerr <<"Error: Wrong value for output type "<<output<<" Must be glfw or netcdf! Exit now!";
#ifdef WITH_MPI
        MPI_Abort(MPI_COMM_WORLD, -1);
#endif //WITH_MPI
        return -1;
    }
    ////////////////////////////////////////////////////////////////////
    t.toc();
    unsigned hour = (unsigned)floor(t.diff()/3600);
    unsigned minute = (unsigned)floor( (t.diff() - hour*3600)/60);
    double second = t.diff() - hour*3600 - minute*60;
    DG_RANK0 std::cout << std::fixed << std::setprecision(2) <<std::setfill('0');
    DG_RANK0 std::cout <<"Computation Time \t"<<hour<<":"<<std::setw(2)<<minute<<":"<<second<<"\n";
    DG_RANK0 std::cout <<"which is         \t"<<t.diff()/itstp/maxout<<"s/step\n";
#ifdef WITH_MPI
    MPI_Finalize();
#endif //WITH_MPI
    return 0;

}
