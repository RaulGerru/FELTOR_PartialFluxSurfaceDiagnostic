#pragma once
#include <functional>
#include "dg/backend/memory.h"
#include "dg/topology/geometry.h"
#include "dg/blas1.h"

namespace dg
{

namespace geo
{

///@addtogroup fluxfunctions
	
	
struct convolution : public aCylindricalFunctor<convolution> 
{ 
	/**
    * @brief It makes a poloidal convolution of the different data points of a 2D X-grid: It cuts the data and makes the poloidal average that corresponds to that point in the grid in a range. 
    *
    * For the moment, it has a cut in the zeta range to make it only in the LCFS, as the computation time is quite expensive.
    * 
    * \f[ f(zeta,eta)= \begin{cases}
	*mean(f(zeta,eta)) from eta-range/2 to eta+range/2 \text{ if } zeta=0 (LCFS) \\
	*0 \text{ else }
	*\end{cases}
	*\f]
    * 
    * 
    * @brief <tt> convolution( data, range, f0, GridX2d) </tt>
    * @tparam HVec
    * @param data (THe data matrix that you want to produce the convolution in)
    * @tparam double
    * @param range (how big the poloidal angle in which you want to produce the convolution, in degrees). This should be related to the repetition of structures in the toroidal average, which must be dependent on the toroidal discretitation, the q factor and the parallel velocity.
    * @tparam RealCurvilinearGridX2d<double>
    * @param GridX2d (THe grid over which the data is plotted and on which you are supposed to do the cut with the Grid_cutter functions)
    * 
    * 
    * 
    * 
    * @note How to use it? dg::evaluate(dg::geo::convolution(transferH2dX, 7.5, f0, gridX2d), gridX2d.grid());  After you have it, you will need to get the data to a simple vector, and not a matrix;
    */

	convolution(HVec conv_transferH2dX, double range, double f0, RealCurvilinearGridX2d<double> gridX2d): Conv_transferH2dX(conv_transferH2dX), Range(range), F0(f0), m_g(gridX2d){}//, m_poloidal_average( m_g.grid(), dg::coo2d::y){}//, POloidal_average(gridX2d.grid(), dg::coo2d::y) {}

		double do_compute(double zeta, double eta) const {
			         
			         //ALL of th paragraph can be defined out of the loop
			dg::Average<dg::HVec > m_poloidal_average( m_g.grid(), dg::coo2d::y);	//define poloidal average
			dg::SparseTensor<dg::HVec> metricX = m_g.metric();
			dg::HVec In_VolX2d = dg::tensor::volume2d( metricX);
			dg::HVec in_VolX2d2 = In_VolX2d; //Definition of the variables that we are going to edit inside of the loop
			dg::HVec Cutted_transferH2dX=Conv_transferH2dX;
			dg::HVec Cutted_transferH2dX2=Conv_transferH2dX;
			dg::HVec Conv_dvdpsip, Conv_t1d, Conv_dvdpsip2, Conv_t1d2;



			dg::HVec cut= dg::evaluate(dg::geo::Grid_cutter2(eta, Range, zeta), m_g.grid()); //Cutting the grid with the range and the LCFS (zeta with the if initial condition)
			dg::blas1::pointwiseDot(In_VolX2d, cut, In_VolX2d); //cut the volume grid to do the partial flux surface integral                                              	                                           				
			m_poloidal_average(In_VolX2d, Conv_dvdpsip, false); //Make the poloidal average of the cutted volume matrix
			dg::blas1::scal(Conv_dvdpsip, 4.*M_PI*M_PI*F0); //Normalized the 1D volume vectors
			
			dg::blas1::pointwiseDot( Cutted_transferH2dX, In_VolX2d, Cutted_transferH2dX); //Cut the data matrix 
			
			m_poloidal_average(Cutted_transferH2dX, Conv_t1d, false); //Poloidal average of the cutted data
			dg::blas1::scal(Conv_t1d, 4*M_PI*M_PI*F0); //Normalize the 1D data vector
			dg::blas1::pointwiseDivide(Conv_t1d, Conv_dvdpsip, Conv_t1d); //Divide the data 1D vector with the Volume to make it simple average and not integral.

			double result;
			dg::HVec ones(Conv_t1d); //We make a ones vector to make a dot product with the data vector and, as the entries of the data re going to be 0 except of the LCFS, we will only get the value in that entry.
			dg::blas1::copy( 1., ones);
			int lim=Conv_t1d.size();
			
			for (int i=0; i<lim; i++) //As we have devided by the cutted dvdpsip, we have entries that are NaN, so we need to make them 0 instead of NaN. That's what this loop makes.
			{ if (Conv_t1d[i]!=Conv_t1d[i]) Conv_t1d[i]=0;} 
			
			result=dg::blas1::dot(ones, Conv_t1d); //Multiply the two vectors to obtain the value at LCFS.

			return result;
			}
			
	private:
	HVec Conv_transferH2dX, VolX2d ;
    double Range;
    double F0;
    RealCurvilinearGridX2d<double> m_g;

	
};

//NOTES ON HOW TO USE THE FUNCTORS/STRUCTURES

//When you define a structure (like convolution2), it needs a constructor to be completely defined.
//A constructor is defined with the same name as the structure, and between brackets, the inputs required to pre define the structure.
//If a constructor is not define, there is always a default constructor defined, which is empty, so you need no input and does not make anything.
//The structure of the constructor is as follow:

//name_of_constructor(equal to the structure)(list of inputs):
//initializer list (where we define the constructors to be used on the private variables. ) {}
//There can be several constructors, but with the same name but different number or kind of inputs.
//Inside of the {} calculations can be done, but no return is going to have.
//To create the constructor in the main code it is done like this: Convolution2 conv(  range, f0, gridX);
//conv is the name that is going to have that constructor 
//It is neccesary before using any function inside of the structure.

//After this, we can define functions with the following structure:
//kind of return( int/double/HVec) name_of_function(whatever)(input list) {WHat the function does;
//return output}

//Anything that wants to be used inside of the structure (in functions or constructures) need to be defined in the private section.
// If some of these structures to be used have no default constructor (as averages), it needs to be defined in the initializer list in the constructor
// 


//EXTRA INFO: If we define a function as operator()(inputlist), it means that to call that function, instead of using the name of the constructor already used:
//like conv.name_function(input for function) it simply works like conv(input for function)


//HVecs are only 1 dimensional, so when they are "2D", they are simply 1D, so if we want to work it as Matrices, we have to define things like this:
//conv2d[i*Nzeta+k] = F1D[k]; with i and k the indeces of the two directions of the grid
// like this: std::vector<HVEc> v(10); I can define a vecor of vectors, which would work as a matrix, and could acces the entries as:
//v[0][1] = 1;

struct Convolution_def
{
	Convolution_def(double range, double f0, RealCurvilinearGridX2d<double> gridX2d, int Neta): 
	m_range(range), m_f0(f0), m_g(gridX2d), m_poloidal_average(m_g.grid(), dg::coo2d::y)
	{	volX2d = dg::tensor::volume2d(m_g.metric()); 
		dg::HVec m_conv_vol1d;
		double d_eta=(m_g.x1()-m_g.x0())/m_g.lx();//is it lx() or Nx()*n()??
    
		for (int eta=m_g.x0(); eta<m_g.x1(); eta+=d_eta) 
			{ 
				dg::blas1::pointwiseDot(volX2d, dg::evaluate(dg::geo::Grid_cutter(eta, m_range), m_g.grid()), volX2d); //cut the volume grid to do the partial flux surface integral                                              
				m_poloidal_average(volX2d , m_conv_vol1d, false);
				dg::blas1::scal( m_conv_vol1d, 4.*M_PI*M_PI*m_f0);
				
				for (int zeta=0; zeta<m_g.lx(); zeta++)
				{
				m_conv_volX2d[eta/d_eta*m_g.lx()+zeta]=m_conv_vol1d[zeta];  //SEGMENTATION FAULT DUE TO m_conv_volX2d, no idea why yet 					
				};
			};
		
	}
	
	HVec convoluted_grid() {return m_conv_volX2d;} //Function to return the convoluted volume function
	
	HVec convolute(const HVec F){
	return m_F_convoluted;
	}//This function will be the convolution function
	
	//I would like to define as another function the partial flux surface average, but when I come back.
	
	private:
	HVec m_F, m_F_convoluted, volX2d, m_convoluted_grid;
    double m_range;
    double m_f0;
    RealCurvilinearGridX2d<double> m_g;
	Average<dg::HVec> m_poloidal_average;
	Grid1d m_g1d_eta; 	
	
	public:
	HVec m_conv_volX2d; //I made this public to try to solve the segmentation fault (didn't work)
};






/*

struct convolution2 //: public aCylindricalFunctor<convolution2> 
{ 
	
    * @brief It makes a poloidal convolution of the different data points of a 2D X-grid: It cuts the data and makes the poloidal average that corresponds to that point in the grid in a range. 
    *
    * For the moment, it has a cut in the zeta range to make it only in the LCFS, as the computation time is quite expensive.
    * 
    * \f[ f(zeta,eta)= \begin{cases}
	*mean(f(zeta,eta)) from eta-range/2 to eta+range/2 \text{ if } zeta=0 (LCFS) \\
	*0 \text{ else }
	*\end{cases}
	*\f]
    * 
    * 
    * @brief <tt> convolution( data, range, f0, GridX2d) </tt>
    * @tparam HVec
    * @param data (THe data matrix that you want to produce the convolution in)
    * @tparam double
    * @param range (how big the poloidal angle in which you want to produce the convolution, in degrees). This should be related to the repetition of structures in the toroidal average, which must be dependent on the toroidal discretitation, the q factor and the parallel velocity.
    * @tparam RealCurvilinearGridX2d<double>
    * @param GridX2d (THe grid over which the data is plotted and on which you are supposed to do the cut with the Grid_cutter functions)
    * 
    * 
    * 
    * 
    * @note How to use it? dg::evaluate(dg::geo::convolution(transferH2dX, 7.5, f0, gridX2d), gridX2d.grid());  After you have it, you will need to get the data to a simple vector, and not a matrix;
    
	//CONSTRUCTOR HAS THE SAME NAME AS THE CLASS/STRUCTURE AND THE FUNCTIONS DO NOT. AND CONSTRUCTOR HAS NO RETURN, WHILE THE FUNCTIONS CAN HAVE
	//Convolution2(){}//default constructor
	Convolution2(double range, double f0, RealCurvilinearGridX2d<double> gridX2d): 
	mm_F(f), m_f0(f0), m_g(gridX2d) //consttruction done
	{
		m_range= range;
		//compute convolution of metric
		std::cout << "Hello world"<<std::endl;
	
		//compute... and store in m_convg
	}//m2_poloidal_average(m_g.grid(), dg::coo2d::y){}
	//Convolution2( const Convolution2& conv);//copy constructor
		HVec operator()(const HVec& F) const {
	//HVec member_function( const HVec & f, dg::Grid1d grid_eta) const{
			//in mmmain
			// HVEc conv4 ( const HVEc & f){ std::cout << "Hahahaha, trick"}
			HVEc v; //calls default constructor
			Convolution2 conv; //compiler error no default constructor
			Convolution2 conv(  range, f0, gridX);
			Convolution2 conv3( conv); //aaaaaalways works
			HVEc result = conv3( vector); // -> conv.operator()(vector);
			HVEc restul2 = conv4(vector);
			HVEc result2 = conv.member_function( veector2);
		
		std::cout<<eta<<"\n";
		 	dg::HVec F1D;//1d vector size m_g.Nx()m_g.n()
			dg::Average<dg::HVec> m2_poloidal_average( m_g.grid(), dg::coo2d::y);	//define poloidal average
			dg::HVec copy_m_F=m_F;
		
			dg::blas1::pointwiseDot(m_F, dg::evaluate(dg::geo::Grid_cutter(eta, m_range), m_g.grid()), copy_m_F); //cut the volume grid to do the partial flux surface integral                                              
	
			m2_poloidal_average( copy_m_F, F1D, false); //allocates space for m_F1D 
			dg::blas1::scal( F1D, 4.*M_PI*M_PI*m_f0
			std::vector<HVEc> v(10);
			v[0] = F1D;
			for(int i=0; i<Neta; i++)
			for( int k=0; k<Nzeta; k++)
			conv2d[i*Nzeta+k] = F1D[k];

	        return F1D;
	       
			}
			
	private:
	HVec m_F, m_conv2d, m_convg;//, m_F1D;
    double m_range;
    double m_f0;
    RealCurvilinearGridX2d<double> m_g;
    //Average<dg::HVec> m2_poloidal_average;
	
};
*/

/*
	dg::HVec convolutionf(const HVec F, const double range, const double f0, const RealCurvilinearGridX2d<double> gridX2d)
	{	int lim=gridX2d.ly();//or F.size();
		int lim2=gridX2d.lx();
		dg::HVec F_1D, copy_F=F;
		dg::HVec result;
		dg::Average<dg::HVec > m_poloidal_average( gridX2d.grid(), dg::coo2d::y);	//define poloidal average
		for (int eta=0; eta<lim; eta++)
		{			
			dg::Average<dg::HVec> m_poloidal_average( gridX2d.grid(), dg::coo2d::y);	//define poloidal average
		
			dg::blas1::pointwiseDot(F, dg::evaluate(dg::geo::Grid_cutter(gridX2d., range), gridX2d.grid()), copy_F); //cut the volume grid to do the partial flux surface integral                                              
	
			m_poloidal_average( copy_F, F_1D, false);  
			dg::blas1::scal( F_1D, 4.*M_PI*M_PI*f0);
			for (int zeta=0; zeta<lim2; zeta++)
			{result[zeta][eta]=F_1D[zeta];}
			
			std::cout<<eta<<"\n";
			
		}
		return result;
		
	}
	

*/






}
}


