
:
	a_1/inputPlaceholder*
dtype0*
shape
:
O
a_1/truncated_normal/shapeConst*
valueB" @   @  *
dtype0
F
a_1/truncated_normal/meanConst*
valueB
 *    *
dtype0
H
a_1/truncated_normal/stddevConst*
valueB
 *  �?*
dtype0
�
$a_1/truncated_normal/TruncatedNormalTruncatedNormala_1/truncated_normal/shape*
T0*
dtype0*
seed2 *

seed 
k
a_1/truncated_normal/mulMul$a_1/truncated_normal/TruncatedNormala_1/truncated_normal/stddev*
T0
Y
a_1/truncated_normalAdda_1/truncated_normal/mula_1/truncated_normal/mean*
T0
]
a_1/w
VariableV2*
dtype0*
shared_name *
	container *
shape:����

a_1/w/AssignAssigna_1/wa_1/truncated_normal*
T0*
use_locking(*
validate_shape(*
_class

loc:@a_1/w
@

a_1/w/readIdentitya_1/w*
T0*
_class

loc:@a_1/w
r
a_1/matrix1/initial_valueConst*A
value8B6
"(��*?�u�<G���r?à3?׉��D/��G���!<��\��?*
dtype0
_
a_1/matrix1
VariableV2*
dtype0*
shared_name *
	container *
shape
:

�
a_1/matrix1/AssignAssigna_1/matrix1a_1/matrix1/initial_value*
T0*
use_locking(*
validate_shape(*
_class
loc:@a_1/matrix1
R
a_1/matrix1/readIdentitya_1/matrix1*
T0*
_class
loc:@a_1/matrix1
�
a_1/matrix2/initial_valueConst*�
value�B�

"�%汾N�+?� �>'*w=ŏ���E��E+�>	�?�E�$Sp�H�N��΅?��	?�O?��v>��s����05�?�1>����5)�?Y�>��:?���?�;�[�-�f ��|>�/?z����[<�o?�[�>Ͽ�>G���:��;�?H����x׾O����k�B>p?�>�ɩ�	�пC~?Q~��7%����?�`�>(IP�� 4?�I��	��eO�?�F-? L@�O��͘'<6+�>���?�����?��.>��5��7�$0���	Ͻ�d�U7�`j�?b�k?}o�?�K��y��a�	@���>���>�CQ�q3�>�;�=��=%��?qE��D>��ܿ+*�>L튿ud�?E���	��?�R�׎̿��g>�o>,}r�T{�����㢪�*
dtype0
_
a_1/matrix2
VariableV2*
dtype0*
shared_name *
	container *
shape
:


�
a_1/matrix2/AssignAssigna_1/matrix2a_1/matrix2/initial_value*
T0*
use_locking(*
validate_shape(*
_class
loc:@a_1/matrix2
R
a_1/matrix2/readIdentitya_1/matrix2*
T0*
_class
loc:@a_1/matrix2
r
a_1/matrix4/initial_valueConst*A
value8B6
"(n�>y����>�6�?��g?`}?|y�Ƨ�>r5��*
dtype0
_
a_1/matrix4
VariableV2*
dtype0*
shared_name *
	container *
shape
:

�
a_1/matrix4/AssignAssigna_1/matrix4a_1/matrix4/initial_value*
T0*
use_locking(*
validate_shape(*
_class
loc:@a_1/matrix4
R
a_1/matrix4/readIdentitya_1/matrix4*
T0*
_class
loc:@a_1/matrix4
a
a_1/matmul1MatMul	a_1/inputa_1/matrix1/read*
T0*
transpose_b( *
transpose_a( 
c
a_1/matmul2MatMula_1/matmul1a_1/matrix2/read*
T0*
transpose_b( *
transpose_a( 
c
a_1/matmul4MatMula_1/matmul2a_1/matrix4/read*
T0*
transpose_b( *
transpose_a( 
^
a_1/initNoOp^a_1/matrix1/Assign^a_1/matrix2/Assign^a_1/matrix4/Assign^a_1/w/Assign"