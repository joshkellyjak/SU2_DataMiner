.. _doc_fgm_tabulation:

.. sectionauthor:: Evert Bunschoten 

Tabulation methods for FGM applications 
=======================================


This page documents the tabulation methods for FGM applications in *SU2 DataMiner*. The FGM table generator class is derived from the :ref:`base table generator class <doc_tablegeneration_base>` from which it inherets most of its functionalities.
FGM tables can be 2D or 3D, depending on the controlling variables specified in the *SU2 DataMiner* configuration. The extra functionalities of the FGM table generator with respect to the base class are :ref:`gradient-based refinement <gradient_based_refinement>`, and additional refinement near :ref:`chemical equilibrium conditions <equilibrium_refinement>`. 

A tutorial which demonstrates the functionalities of the FGM table generator for a methane-air application can be found :ref:`here <tutorial_methane_tabulation>`. A tutorial for hydrogen-based tabulation will come in a later update.

.. contents:: :depth: 2

The number of table dimensions and the controlling variables are retrieved from the *SU2 DataMiner* configuration upon initialization. 

.. autofunction:: Manifold_Generation.LUT.LUTGenerators.SU2TableGenerator_FGM.__init__


.. _gradient_based_refinement:

Gradient-based refinement 
-------------------------

Thermochemical state variables such as source terms can have very high gradients with respect to the progress variable. In order to accurately evaluate such quantities with the table, the resolution should scale with the local magnitude of the gradients rather than being based on the values of thermochemical state variables.
The following function can be used to scale the local mesh resolution based on the local value of gradients of specified quantities. 

.. autofunction:: Manifold_Generation.LUT.LUTGenerators.SU2TableGenerator_FGM.applyRefinementForGradientOf

The refinement factor :math:`r` applied at an arbitrary location in the table :math:`X^*` based on the gradient of quantity :math:`j` is evaluated as 

.. math::

    r_j(X^*) = f_j\frac{\|\nabla \Psi_j(X^*)\|}{\max \|\nabla \Psi_j\|},

in which :math:`f_j` is the value of the coefficient supplied by the function argument. The refinement factor therefore scales linearly with the gradient of quantity :math:`j`, normalized by the global maximum value of the gradient :math:`\max \|\nabla \Psi_j\|`.
The gradients are approximated with central finite-differences

.. math::

    \nabla \Psi_j(X^*) = \frac{\Psi_j(X^* + \delta) - \Psi_j(X^* - \delta)}{2\delta},

where the step size :math:`\delta` is currently hard-coded to a value of 1e-2.

.. important::

    Evaluating the gradients with finite-differences is computationally costly. Adding a gradient-based refinement will therefore significantly increase the computational cost of table generation.


.. _equilibrium_refinement:

Extra refinement around chemical equilibrium 
--------------------------------------------

For the initialization of premixed flames in SU2 FGM simulations, it is very important that the total enthalpy can be accurately retrieved through inverse regression. 
Additional refinement can be imposed at the sections of the table where the mixture is in chemical equilibrium using the following function:

.. autofunction:: Manifold_Generation.LUT.LUTGenerators.SU2TableGenerator_FGM.refineEquilibrium

Here, the local grid refinement factor is calculated with the following function

.. math::

    r(X^*) = f\left(H(c^* - c^*_R + m) + H(c^*_R + m - c^*)\right),

where :math:`c^*` is the local, scaled value of the progress variable, :math:`c^*_R` the value of the progress variable for stable, premixed reactants, :math:`c^*_P` that of equilibrated reaction products, and :math:`m` an additional margin whithin which the refinement is applied.
