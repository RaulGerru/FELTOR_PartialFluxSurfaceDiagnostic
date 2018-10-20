#include <iostream>
#include <iomanip>
#include "json/json.h"

#include "mpi.h"
#undef DG_DEBUG
#include "dg/backend/timer.h"
#include "dg/backend/mpi_init.h"
#include "dg/geometry/functions.h"
#include "dg/blas.h"
#include "dg/functors.h"
#include "dg/geometry/geometry.h"
#include "testfunctors.h"
#include "ds.h"
#include "solovev.h"
#include "flux.h"
#include "toroidal.h"
#include "mpi_curvilinear.h"


int main(int argc, char * argv[])
{
    MPI_Init( &argc, &argv);
    int rank;
    unsigned n, Nx, Ny, Nz, mx[2];
    MPI_Comm comm;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank);
    if(rank==0)std::cout << "# Test DS on flux grid!"<<std::endl;
    dg::mpi_init3d( dg::DIR, dg::PER, dg::PER, n, Nx, Ny, Nz, comm);
    if( rank == 0)
    {
        std::cout <<"# You typed\n"
                  <<"n:  "<<n<<"\n"
                  <<"Nx: "<<Nx<<"\n"
                  <<"Ny: "<<Ny<<"\n"
                  <<"Nz: "<<Nz<<std::endl;
        std::cout <<"# Type mx (1) and my (100)\n";
        std::cin >> mx[0] >> mx[1];
        std::cout << "# You typed\n"
                  <<"mx: "<<mx[0]<<"\n"
                  <<"my: "<<mx[1]<<std::endl;
        std::cout << "# Create parallel Derivative!\n";
    }
    MPI_Bcast( mx, 2, MPI_INT, 0, MPI_COMM_WORLD);
    Json::Value js;
    if( argc==1)
    {
        std::ifstream is("geometry_params_Xpoint.js");
        is >> js;
    }
    else
    {
        std::ifstream is(argv[1]);
        is >> js;
    }
    dg::geo::solovev::Parameters gp(js);
    dg::geo::TokamakMagneticField mag = dg::geo::createSolovevField( gp);
    double psi_0 = -20, psi_1 = -4;
    dg::Timer t;
    t.tic();
    dg::geo::FluxGenerator flux( mag.get_psip(), mag.get_ipol(), psi_0, psi_1, gp.R_0, 0., 1);
    if(rank==0)std::cout << "# Constructing Grid..."<<std::endl;
    dg::geo::CurvilinearProductMPIGrid3d g3d(flux, n, Nx, Ny,Nz, dg::NEU, dg::PER, dg::PER, comm);
    if(rank==0)std::cout << "# Constructing Fieldlines..."<<std::endl;
    dg::geo::DS<dg::aProductMPIGeometry3d, dg::MIDMatrix, dg::MDMatrix, dg::MDVec> ds( mag, g3d, dg::NEU, dg::PER, dg::geo::FullLimiter(), dg::centered, 1e-8, mx[0], mx[1]);

    t.toc();
    if(rank==0)std::cout << "# Construction took "<<t.diff()<<"s\n";
    ///##########################################################///
    //apply to function (MIND THE PULLBACK!)
    const dg::MDVec function = dg::pullback( dg::geo::TestFunctionPsi2(mag), g3d);
    dg::MDVec derivative(function);
    dg::MDVec sol0 = dg::pullback( dg::geo::DsFunction<dg::geo::TestFunctionPsi2>(mag), g3d);
    dg::MDVec sol1 = dg::pullback( dg::geo::DssFunction<dg::geo::TestFunctionPsi2>(mag), g3d);
    dg::MDVec sol2 = dg::pullback( dg::geo::DsDivFunction<dg::geo::TestFunctionPsi2>(mag), g3d);
    dg::MDVec sol3 = dg::pullback( dg::geo::DsDivDsFunction<dg::geo::TestFunctionPsi2>(mag), g3d);
    std::vector<std::pair<std::string, const dg::MDVec&>> names{
         {"forward",sol0}, {"backward",sol0},
         {"centered",sol0}, {"dss",sol1},
         {"forwardDiv",sol2}, {"backwardDiv",sol2}, {"centeredDiv",sol2},
         {"forwardLap",sol3}, {"backwardLap",sol3}, {"centeredLap",sol3}
    };
    ///##########################################################///
    if(rank==0)std::cout << "# TEST Flux (No Boundary conditions)!\n";
    if(rank==0)std::cout <<"Flux:\n";
    const dg::MDVec vol3d = dg::create::volume( g3d);
    for( const auto& name :  names)
    {
        callDS( ds, name.first, function, derivative);
        dg::blas1::axpby( 1., name.second, -1., derivative);
        double sol = dg::blas2::dot( vol3d, name.second);
        double norm = dg::blas2::dot( derivative, vol3d, derivative);
        if(rank==0)std::cout <<"    "<<name.first<<":"
                  <<std::setw(16-name.first.size())
                  <<" "<<sqrt(norm/sol)<<"\n";
    }
    ///##########################################################///
    std::vector<std::pair<std::string, dg::direction>> namesLap{
         {"invForwardLap",dg::forward}, {"invBackwardLap",dg::backward}, {"invCenteredLap",dg::centered}
    };
    dg::MDVec solution = dg::pullback( dg::geo::OMDsDivDsFunction<dg::geo::TestFunctionPsi2>(mag), g3d);
    dg::Invert<dg::MDVec> invert( solution, g3d.size(), 1e-10);
    dg::geo::TestInvertDS< dg::geo::DS<dg::aProductMPIGeometry3d, dg::MIDMatrix, dg::MDMatrix, dg::MDVec>, dg::MDVec>
        rhs(ds);
    for( auto name : namesLap)
    {
        ds.set_direction( name.second);
        invert( rhs, derivative, solution);
        dg::blas1::axpby( 1., function, -1., derivative);
        double sol = dg::blas2::dot( vol3d, function);
        double norm = dg::blas2::dot( derivative, vol3d, derivative);
        if(rank==0)std::cout <<"    "<<name.first<<":"
                  <<std::setw(16-name.first.size())
                  <<" "<<sqrt(norm/sol)<<"\n";
    }
    MPI_Finalize();
    return 0;
}
