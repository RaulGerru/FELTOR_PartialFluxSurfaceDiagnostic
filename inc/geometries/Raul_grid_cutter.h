#pragma once
#include <functional>
#include "dg/backend/memory.h"
#include "dg/topology/geometry.h"

namespace dg
{
namespace geo
{

struct Grid_cutter : public aCylindricalFunctor<Grid_cutter>
{

    Grid_cutter(double eta_0, double eta_size): eta0(eta_0), etasize(eta_size){} //eta_0 is in radians and eta_size is in degrees
    //BETTER DOCUMENTATION DESCRIPTION WHEN I CAN SEE HOW IT WOULD LOOK IN THE DOCUMENTATION
    
    double do_compute(double zeta, double eta) const {
	double eta_up_lim=eta0+etasize*M_PI/(2*180);
    double eta_down_lim=eta0-etasize*M_PI/(2*180);
    
    if (eta_up_lim>2*M_PI) {		
		eta_up_lim+=-2*M_PI;
        if( (eta<eta_up_lim || eta>eta_down_lim))
            return 1;
        return 0;
	}
    if (eta_down_lim<0)  {
		eta_down_lim+=2*M_PI;
        if( (eta<eta_up_lim || eta>eta_down_lim))
            return 1;
        return 0;   
	}
    else
    {
        if( eta<eta_up_lim && eta>eta_down_lim)
            return 1;
        return 0;
	}
    }
    private:
    double eta0, etasize;
}; 

struct Grid_cutter2 : public aCylindricalFunctor<Grid_cutter2>
{

    Grid_cutter2(double eta_0, double eta_size, double zeta_def): eta0(eta_0), etasize(eta_size), zetadef(zeta_def){} //eta_0 is in radians and eta_size is in degrees
    //BETTER DOCUMENTATION DESCRIPTION WHEN I CAN SEE HOW IT WOULD LOOK IN THE DOCUMENTATION


    double do_compute(double zeta, double eta) const {
	double eta_up_lim=eta0+etasize*M_PI/(2*180);
    double eta_down_lim=eta0-etasize*M_PI/(2*180);
    
    if (eta_up_lim>2*M_PI) {		
		eta_up_lim+=-2*M_PI;
        if( (eta<eta_up_lim || eta>eta_down_lim) && zeta==zetadef)
            return 1;
        return 0;
	}
    if (eta_down_lim<0)  {
		eta_down_lim+=2*M_PI;
        if( (eta<eta_up_lim || eta>eta_down_lim) && zeta==zetadef)
            return 1;
        return 0;   
	}
    else
    {
        if( eta<eta_up_lim && eta>eta_down_lim && zeta==zetadef)
            return 1;
        return 0;
	}
}
    private:
    double eta0, etasize, zetadef;
};




/*
struct Grid_cutter : public aCylindricalFunctor<Grid_cutter>
{

    Grid_cutter(double eta_0, double eta_size, double max_psi, double min_psi): eta0(eta_0), etasize(eta_size), Psi_max(max_psi), Psi_min(min_psi){} //eta_0 is in radians and eta_size is in degrees
    //BETTER DOCUMENTATION DESCRIPTION WHEN I CAN SEE HOW IT WOULD LOOK IN THE DOCUMENTATION
    
    double do_compute(double zeta, double eta) const {
	double eta_up_lim=eta0+etasize*M_PI/(2*180);
    double eta_down_lim=eta0-etasize*M_PI/(2*180);
    
    if (eta_up_lim>2*M_PI) {		
		eta_up_lim+=-2*M_PI;
        if( (eta<eta_up_lim || eta>eta_down_lim))// && zeta>Psi_min && zeta<Psi_max) 
            return 1;
        return 0;
	}
    if (eta_down_lim<0)  {
		eta_down_lim+=2*M_PI;
        if( (eta<eta_up_lim || eta>eta_down_lim))// && zeta>Psi_min && zeta<Psi_max)
            return 1;
        return 0;   
	}
    else
    {
        if( eta<eta_up_lim && eta>eta_down_lim)// && zeta>Psi_min && zeta<Psi_max)
            return 1;
        return 0;
	}
    }
    private:
    double eta0, etasize, Psi_max, Psi_min ;
};
*/

struct PsiCutter : public aCylindricalFunctor<PsiCutter>
{
    PsiCutter(double Psilim): Psi_lim(Psilim){}
    double do_compute(double Psi) const {
        if( Psi > Psi_lim && Psi<0)
            return 1;
        return 0;
    }
    private:
    double Psi_lim;
};

};//namespace geo
}//namespace dg



