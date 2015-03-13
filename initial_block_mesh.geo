/********************************************************************* 
 *
 *  Simple block mesh
 *
 *  Run this command to make a .msh file
 *  gmsh initial_block_mesh.geo -3 -optimize
 *
 *  Use dolfin-convert file.msh file.xml to create an xml file
 *  for dolfin.
 *
 *********************************************************************/

// Number of elements on each side
N = 10;

// Affectation
lc = 1.0/N;

// dimensions of the box
Point(1) = {-1.55, -1.55, -0.05, lc};
Point(2) = {1.55, -1.55,  -0.05, lc} ;
Point(3) = {1.55, 1.55, -0.05, lc} ;
Point(4) = {-1.55, 1.55, -0.05, lc} ;

Line(1) = {1,2} ;
Line(2) = {3,2} ;
Line(3) = {3,4} ;
Line(4) = {4,1} ;

// Define loop of lines at the bottom of the box
Line Loop(5) = {4,1,-2,3} ;

// Define surface from line loop
Plane Surface(6) = {5} ;

// Extend in z direction

Extrude {0, 0, 1.10} { Surface{6}; }


// To save all the tetrahedra discretizing the volumes 129 and 130
// with a common region number, we finally define a physical
// volume:

//Physical Volume (1) = {129,130};

