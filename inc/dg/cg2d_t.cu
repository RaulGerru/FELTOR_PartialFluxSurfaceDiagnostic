#include <iostream>
#include <iomanip>

#include "cg.h"
#include "elliptic.h"

const double lx = 2.*M_PI;
const double ly = 2.*M_PI;

const double eps_ = 1e-6; //# of pcg iterations increases very much if
 // eps << relativer Abstand der exakten Lösung zur Diskretisierung vom Sinus

double fct(double x, double y){ return sin(y)*sin(x);}
double laplace_fct( double x, double y) { return 2*sin(y)*sin(x);}
double initial( double x, double y) {return sin(0);}

int main()
{
    //global relative error in L2 norm is O(h^P)
    //more N means less iterations for same error
    unsigned n, Nx, Ny;
    std::cout << "Type n, Nx and Ny! \n";
    std::cin >> n >> Nx >> Ny;
    std::cout << "Computing on the Grid " <<n<<" x "<<Nx<<" x "<<Ny <<std::endl;
    dg::Grid2d grid( 0, lx, 0, ly,n, Nx, Ny, dg::PER, dg::PER);
    const dg::HVec w2d = dg::create::weights( grid);
    const dg::HVec v2d = dg::create::inv_weights( grid);
    std::cout<<"Evaluate initial condition\n";
    dg::HVec x = dg::evaluate( initial, grid);

//! [doxygen]
    std::cout << "Create Laplacian\n";
    dg::Elliptic<dg::CartesianGrid2d, dg::HMatrix, dg::HVec> A( grid);
    dg::CG<dg::HVec > pcg( x, n*n*Nx*Ny);
    std::cout<<"Evaluate right hand side\n";
    dg::HVec b = dg::evaluate ( laplace_fct, grid);
    const dg::HVec solution = dg::evaluate ( fct, grid);
    dg::blas2::symv( w2d, b, b);
    std::cout << "Number of pcg iterations "<< pcg( A, x, b, v2d, eps_)<<std::endl;
//! [doxygen]
    //std::cout << "Number of cg iterations "<< pcg( A, x, b, dg::Identity<double>(), eps)<<std::endl;
    std::cout << "For a precision of "<< eps_<<std::endl;
    //compute error
    dg::HVec error( solution);
    dg::blas1::axpby( 1.,x,-1.,error);

    dg::HVec Ax(x), resi( b);
    dg::blas2::symv(  A, x, Ax);
    dg::blas1::axpby( 1.,Ax,-1.,resi);

    exblas::udouble res;
    res.d = sqrt(dg::blas2::dot( w2d, x));
    std::cout << "L2 Norm of x0 is              " << res.d<<"\t"<<res.i << std::endl;
    res.d = sqrt(dg::blas2::dot(w2d , solution));
    std::cout << "L2 Norm of Solution is        " << res.d<<"\t"<<res.i << std::endl;
    res.d = sqrt(dg::blas2::dot(w2d , error));
    std::cout << "L2 Norm of Error is           " << res.d<<"\t"<<res.i << std::endl;
    res.d = sqrt(dg::blas2::dot( w2d, resi));
    std::cout << "L2 Norm of Residuum is        " << res.d<<"\t"<<res.i << std::endl;
    //Fehler der Integration des Sinus ist vernachlässigbar (vgl. evaluation_t)

    return 0;
}
