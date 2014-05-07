#ifndef _DG_LAPLACE_CUH
#define _DG_LAPLACE_CUH

#include <cusp/coo_matrix.h>
#include <cusp/transpose.h>
#include <cusp/elementwise.h>

#include "grid.cuh"
#include "functions.h"
#include "operator_dynamic.h"
#include "creation.cuh"
#include "dx.cuh"
#include "operator_matrix.cuh"

/*! @file 1d laplacians
  */

namespace dg
{

namespace create{
///@cond
/**
 * @brief Create and assemble a cusp Matrix for the negative periodic 1d laplacian in LSPACE
 *
 * @ingroup highlevel
 * Use cusp internal conversion to create e.g. the fast ell_matrix format.
 * @tparam T value-type * @param n Number of Legendre nodes per cell
 * @param N Vector size ( number of cells)
 * @param h cell size
 * @param no use normed if you want to compute e.g. diffusive terms
             use not_normed if you want to solve symmetric matrix equations (T is missing)
 * @param alpha Optional parameter for penalization term
 *
 * @return Host Matrix in coordinate form 
 */
template< class T>
cusp::coo_matrix<int, T, cusp::host_memory> laplace1d_per( unsigned n, unsigned N, T h, norm no = not_normed, T alpha = 1.)
{
//DEPRECATED
    if( n ==1 ) alpha = 0; //makes laplacian of order 2
    cusp::coo_matrix<int, T, cusp::host_memory> A( n*N, n*N, 3*n*n*N);
    //std::cout << A.row_indices.size(); 
    //std::cout << A.num_cols; //this works!!
    Operator<T> l = create::lilj(n);
    Operator<T> r = create::rirj(n);
    Operator<T> lr = create::lirj(n);
    Operator<T> rl = create::rilj(n);
    Operator<T> d = create::pidxpj(n);
    Operator<T> t = create::pipj_inv(n);
    t *= 2./h;
    Operator< T> a = lr*t*rl+(d+l)*t*(d+l).transpose() + alpha*(l+r);
    Operator< T> b = -((d+l)*t*rl+alpha*rl);
    Operator< T> bT = b.transpose();
    if( no == normed) { a = t*a; b = t*b; bT = t*bT; }
    //std::cout << a << "\n"<<b <<std::endl;
    //assemble the matrix
    int number = 0;
    for( unsigned k=0; k<n; k++)
    {
        for( unsigned l=0; l<n; l++)
            detail::add_index<T>(n, A, number, 0,0,k,l, a(k,l)); //1 x A
        for( unsigned l=0; l<n; l++)
            detail::add_index<T>(n, A, number, 0,1,k,l, b(k,l)); //1+ x B
        for( unsigned l=0; l<n; l++)
            detail::add_index<T>(n, A, number, 0,N-1,k,l, bT(k,l)); //1- x B^T
    }
    for( unsigned i=1; i<N-1; i++)
        for( unsigned k=0; k<n; k++)
        {
            for( unsigned l=0; l<n; l++)
                detail::add_index<T>(n, A, number, i, i-1, k, l, bT(k,l));
            for( unsigned l=0; l<n; l++)
                detail::add_index<T>(n, A, number, i, i, k, l, a(k,l));
            for( unsigned l=0; l<n; l++)
                detail::add_index<T>(n, A, number, i, i+1, k, l, b(k,l));
        }
    for( unsigned k=0; k<n; k++)
    {
        for( unsigned l=0; l<n; l++) 
            detail::add_index<T>(n, A, number, N-1,0,  k,l, b(k,l));
        for( unsigned l=0; l<n; l++)
            detail::add_index<T>(n, A, number, N-1,N-2,k,l, bT(k,l));
        for( unsigned l=0; l<n; l++)
            detail::add_index<T>(n, A, number, N-1,N-1,k,l, a(k,l));
    }
    return A;
};
/**
 * @brief Create and assemble a cusp Matrix for the Dirichlet negative 1d laplacian in LSPACE
 *
 * @ingroup highlevel
 * Use cusp internal conversion to create e.g. the fast ell_matrix format.
 * @tparam T value-type
 * @param n Number of Legendre nodes per cell
 * @param N Vector size ( number of cells)
 * @param h cell size
 * @param no use normed if you want to compute e.g. diffusive terms
             use not_normed if you want to solve symmetric matrix equations (T is missing)
 *
 * @return Host Matrix in coordinate form 
 */
template< class T>
cusp::coo_matrix<int, T, cusp::host_memory> laplace1d_dir( unsigned n, unsigned N, T h, norm no = not_normed)
{
//DEPRECATED
    //if( n == 1) alpha = 0; //not that easily because dirichlet 
    cusp::coo_matrix<int, T, cusp::host_memory> A( n*N, n*N, 3*n*n*N - 2*n*n);
    Operator<T> l = create::lilj(n);
    Operator<T> r = create::rirj(n);
    Operator<T> lr = create::lirj(n);
    Operator<T> rl = create::rilj(n);
    Operator<T> d = create::pidxpj(n);
    Operator<T> s = create::pipj(n);
    Operator<T> t = create::pipj_inv(n);
    t *= 2./h;

    Operator<T> a = lr*t*rl+(d+l)*t*(d+l).transpose() + (l+r);
    Operator<T> b = -((d+l)*t*rl+rl);
    Operator<T> bT= b.transpose();
    Operator<T> ap = d*t*d.transpose() + (l + r);
    Operator<T> bp = -(d*t*rl + rl);
    Operator<T> bpT= bp.transpose();
    if( no == normed) { 
        a=t*a; b=t*b; bT=t*bT; 
        ap=t*ap; bp=t*bp; bpT=t*bpT;
    }
    //assemble the matrix
    int number = 0;
    for( unsigned k=0; k<n; k++)
    {
        for( unsigned l=0; l<n; l++)
            detail::add_index<T>(n, A, number, 0,0,k,l, ap(k,l));
        for( unsigned l=0; l<n; l++)
            detail::add_index<T>(n, A, number, 0,1,k,l, bp(k,l));
    }
    for( unsigned k=0; k<n; k++)
    {
        for( unsigned l=0; l<n; l++)
            detail::add_index<T>(n, A, number, 1, 1-1, k, l, bpT(k,l));
        for( unsigned l=0; l<n; l++)
            detail::add_index<T>(n, A, number, 1, 1, k, l, a(k,l));
        for( unsigned l=0; l<n; l++)
            detail::add_index<T>(n, A, number, 1, 1+1, k, l, b(k,l));
    }
    for( unsigned i=2; i<N-1; i++)
        for( unsigned k=0; k<n; k++)
        {
            for( unsigned l=0; l<n; l++)
                detail::add_index<T>(n, A, number, i, i-1, k, l, bT(k,l));
            for( unsigned l=0; l<n; l++)
                detail::add_index<T>(n, A, number, i, i, k, l, a(k,l));
            for( unsigned l=0; l<n; l++)
                detail::add_index<T>(n, A, number, i, i+1, k, l, b(k,l));
        }
    for( unsigned k=0; k<n; k++)
    {
        for( unsigned l=0; l<n; l++)
            detail::add_index<T>(n, A, number, N-1,N-2,k,l, bT(k,l));
        for( unsigned l=0; l<n; l++)
            detail::add_index<T>(n, A, number, N-1,N-1,k,l, a(k,l));
    }
    return A;
}

/*
template< class T>
cusp::coo_matrix<int, T, cusp::host_memory> laplace1d( const Grid1d<T>& g, norm no = not_normed)
{

    if( g.bcx() == DIR)
        return laplace1d_dir<T>( g.n(), g.N(), g.h(), no);
    else 
        return laplace1d_per<T>( g.n(), g.N(), g.h(), no);
}
*/

/**
 * @brief Function for the creation of a 1d laplacian in LSPACE
 *
 * @ingroup highlevel
 * @tparam T value_type
 * @param g The grid on which to create the laplacian (including boundary condition)
 * @param no use normed if you want to compute e.g. diffusive terms
            use not_normed if you want to solve symmetric matrix equations (T is missing)
 *
 * @return Host Matrix in coordinate form
 */
template< class value_type>
cusp::coo_matrix<int, value_type, cusp::host_memory> laplace1d( const Grid1d<value_type>& g, norm no = not_normed, direction dir = forward )
{
    typedef cusp::coo_matrix<int, value_type, cusp::host_memory> HMatrix;
    HMatrix S = dg::tensor( g.N(), dg::create::pipj( g.n())); 
    cusp::blas::scal( S.values, g.h()/2.);
    HMatrix T = dg::tensor( g.N(), dg::create::pipj_inv( g.n())); 
    cusp::blas::scal( T.values, 2./g.h());
    HMatrix right;
    if( dir == forward)
        right = create::dx_plus_mt( g.n(), g.N(), g.h(), g.bcx());
    else if ( dir == backward) 
        right = create::dx_minus_mt( g.n(), g.N(), g.h(), g.bcx());
    else //dir == symmetric
    {
        if( g.bcx() == PER || g.bcx() == NEU_DIR)
            return laplace1d( g, no, forward); //per is symmetric, NEU_DIR cannot be
        if( g.bcx() == DIR_NEU)
            return laplace1d( g, no, backward);//cannot be symmetric
        HMatrix laplus = laplace1d( g, no, forward); //recursive call
        HMatrix laminus = laplace1d( g, no, backward);
        HMatrix laplace;
        cusp::add( laplus, laminus, laplace);
        for( unsigned i=0; i<laplace.values.size(); i++)
            laplace.values[i] *= 0.5;

        return laplace;
    }
    HMatrix left, temp;
    cusp::transpose( right, left);
    cusp::multiply( left, S, temp);

    HMatrix laplace_oJ, laplace;
    cusp::multiply( temp, right, laplace_oJ);
    if( g.n() == 1 && g.bcx() == dg::PER)
    {
        if( no == normed) 
        {
            cusp::multiply( T, laplace_oJ, laplace);
            return laplace;
        }
        return laplace_oJ;
    }
    HMatrix J = dg::create::jump_ot<value_type>( g.n(), g.N(), g.bcx());
    cusp::add( laplace_oJ, J, laplace);
    laplace.sort_by_row_and_column();
    if( no == normed) 
    {
        cusp::multiply( T, laplace, laplace_oJ);
        return laplace_oJ;
    }
    return laplace;
}
///@endcond
} //namespace create

} //namespace dg

#endif // _DG_LAPLACE_CUH
