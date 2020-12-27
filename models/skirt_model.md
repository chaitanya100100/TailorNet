TailorNet garment model assumes that the garment template is a submesh of SMPL
body template, thereby it can take skinning weights from SMPL model.
But that doesn't hold up for garments like skirt.
First, we tried attaching a skirt template to the root joint of SMPL where the skirt
is only subjected to the global rotation of SMPL pose.
All other deformations are learned as displacements in a reasonable manner as shown
in our paper.
However, this design doesn't use the articulation of underlined body parts like legs and
hence it is limiting.

Hence, we propose a simple modification to our garment model which works well for skirt.
We drape a simple skirt template on canonical body and calculate skinning weights of
each skirt vertex as a weighted sum of SMPL skinning weights of K=100 nearest body vertices
where weights of sum are inversely proportional to distances.
This creates a primitive smooth skirt base which is articulated by SMPL pose and shape, and
over which the displacements can be added to predict realistic skirt.
Associating each skirt vertex with K=100 nearest body vertices instead of just one, reduces
the sudden discontinuity of association between two legs.
