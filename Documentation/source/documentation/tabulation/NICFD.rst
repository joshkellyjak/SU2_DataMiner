.. _doc_nicfd_tabulation:

.. sectionauthor:: Evert Bunschoten 

Tabulation methods for NICFD applications 
=========================================


This page documents the tabulation methods for NICFD applications in *SU2 DataMiner* 

.. contents:: :depth: 2


.. autofunction:: Manifold_Generation.LUT.LUTGenerators.SU2TableGenerator_NICFD.__init__ 


.. _doc_nicfd_tabulation_refinement:

Refinement settings 
-------------------

The following methods can be used to specify the table resolution for adaptive tables.
The input value corresponds to the **length scale** of the coarse and refined cells, relative to the scaled thermodynamic state space.
Therefore, a length scale of 0.1 would result in the look-up table to be populated by approximately 100 elements. 

.. autofunction:: Manifold_Generation.LUT.LUTGenerators.SU2TableGenerator_NICFD.setMaximumCellSize
.. autofunction:: Manifold_Generation.LUT.LUTGenerators.SU2TableGenerator_NICFD.applyRefinementWithin


.. _doc_nicfd_tabulation_table_generation:

Table Generation 
----------------

.. autofunction:: Manifold_Generation.LUT.LUTGenerators.SU2TableGenerator_NICFD.setTableVars 
.. autofunction:: Manifold_Generation.LUT.LUTGenerators.SU2TableGenerator_NICFD.generateTable 

Output of tabulation files 
--------------------------

.. autofunction:: Manifold_Generation.LUT.LUTGenerators.SU2TableGenerator_NICFD.writeSU2Table

.. autofunction:: Manifold_Generation.LUT.LUTGenerators.SU2TableGenerator_NICFD.writeParaviewTable

