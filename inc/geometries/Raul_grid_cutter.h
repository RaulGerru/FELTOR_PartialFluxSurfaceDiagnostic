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
    Grid_cutter(double eta_0, double eta_size, double max_psi, double min_psi): eta0(eta_0), etasize(eta_size), Psi_max(max_psi), Psi_min(min_psi){}
    double do_compute(double eta, double zeta) const { //angle, radius		
        if( eta<eta0+etasize*M_PI/(2*180) && eta>eta0+etasize*M_PI/(2*180) && zeta>Psi_min && zeta<Psi_max)
            return 1;
        return 0;
    }
    private:
    double eta0, etasize, Psi_max, Psi_min ;
};

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



