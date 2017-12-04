#pragma once
#include "mpi_vector_blas.h"
#include "mpi_precon.h"
#include "thrust_matrix_blas.cuh"

///@cond
namespace dg
{
namespace blas2
{
namespace detail
{
template< class Precon, class Vector>
inline exblas::Superaccumulator doDot_dispatch( const Vector& x, const Precon& P, const Vector& y, MPIPreconTag, MPIVectorTag)
{
#ifdef DG_DEBUG
    int compare;
    MPI_Comm_compare( x.communicator(), y.communicator(), &compare);
    assert( compare == MPI_CONGRUENT || compare == MPI_IDENT);
    MPI_Comm_compare( x.communicator(), P.communicator(), &compare);
    assert( compare == MPI_CONGRUENT || compare == MPI_IDENT);
#endif //DG_DEBUG
    //local compuation
    exblas::Superaccumulator acc = doDot_dispatch(x.data(), P.data(), y.data(), ThrustMatrixTag(), ThrustVectorTag());
    acc.Normalize();
    //communication
    std::vector<int64_t> result(acc.get_f_words() + acc.get_e_words(), 0);
    MPI_Allreduce(&(acc.get_accumulator()[0]), &(result[0]), acc.get_f_words() + acc.get_e_words(), MPI_LONG, MPI_SUM, x.communicator()); 
    exblas::Superaccumulator acc_fin(result);
    return acc_fin;
}
template< class Precon, class Vector>
inline typename MatrixTraits<Precon>::value_type doDot( const Vector& x, const Precon& P, const Vector& y, MPIPreconTag, MPIVectorTag)
{
    exblas::Superaccumulator acc_fin(doDot_dispatch( x,P,y,MPIPreconTag(),MPIVectorTag()));
    return acc_fin.Round();
}

template< class Matrix, class Vector>
inline typename MatrixTraits<Matrix>::value_type doDot( const Matrix& m, const Vector& x, dg::MPIPreconTag, dg::MPIVectorTag)
{
    exblas::Superaccumulator acc_fin(doDot_dispatch( x,m,x,MPIPreconTag(),MPIVectorTag()));
    return acc_fin.Round();
    //return doDot( x,m,x, MPIPreconTag(), MPIVectorTag());
}

template< class Precon, class Vector>
inline void doSymv(  
              typename MatrixTraits<Precon>::value_type alpha, 
              const Precon& P,
              const Vector& x, 
              typename MatrixTraits<Precon>::value_type beta, 
              Vector& y, 
              MPIPreconTag,
              MPIVectorTag)
{
    doSymv( alpha, P.data(), x.data(), beta, y.data(), ThrustMatrixTag(), ThrustVectorTag());
}

template< class Matrix, class Vector>
inline void doSymv( const Matrix& m, const Vector&x, Vector& y, MPIPreconTag, MPIVectorTag, MPIVectorTag  )
{
    doSymv( 1., m, x, 0, y, MPIPreconTag(), MPIVectorTag());
}


} //namespace detail
} //namespace blas2
} //namespace dg
///@endcond
