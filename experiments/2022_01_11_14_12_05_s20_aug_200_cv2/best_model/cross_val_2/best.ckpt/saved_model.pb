??	
??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
,
Exp
x"T
y"T"
Ttype:

2
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ?
@
StaticRegexFullMatch	
input

output
"
patternstring
?
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.5.02v2.5.0-rc3-213-ga4dfb8d1a718??
`
beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namebeta_1
Y
beta_1/Read/ReadVariableOpReadVariableOpbeta_1*
_output_shapes
: *
dtype0
`
beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namebeta_2
Y
beta_2/Read/ReadVariableOpReadVariableOpbeta_2*
_output_shapes
: *
dtype0
^
decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedecay
W
decay/Read/ReadVariableOpReadVariableOpdecay*
_output_shapes
: *
dtype0
n
learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namelearning_rate
g
!learning_rate/Read/ReadVariableOpReadVariableOplearning_rate*
_output_shapes
: *
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
?
wb_estimator/dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:		**
shared_namewb_estimator/dense/kernel
?
-wb_estimator/dense/kernel/Read/ReadVariableOpReadVariableOpwb_estimator/dense/kernel*
_output_shapes

:		*
dtype0
?
wb_estimator/dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*(
shared_namewb_estimator/dense/bias

+wb_estimator/dense/bias/Read/ReadVariableOpReadVariableOpwb_estimator/dense/bias*
_output_shapes
:	*
dtype0
?
wb_estimator/dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:		*,
shared_namewb_estimator/dense_1/kernel
?
/wb_estimator/dense_1/kernel/Read/ReadVariableOpReadVariableOpwb_estimator/dense_1/kernel*
_output_shapes

:		*
dtype0
?
wb_estimator/dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:	**
shared_namewb_estimator/dense_1/bias
?
-wb_estimator/dense_1/bias/Read/ReadVariableOpReadVariableOpwb_estimator/dense_1/bias*
_output_shapes
:	*
dtype0
?
wb_estimator/dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:	*,
shared_namewb_estimator/dense_2/kernel
?
/wb_estimator/dense_2/kernel/Read/ReadVariableOpReadVariableOpwb_estimator/dense_2/kernel*
_output_shapes

:	*
dtype0
?
wb_estimator/dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_namewb_estimator/dense_2/bias
?
-wb_estimator/dense_2/bias/Read/ReadVariableOpReadVariableOpwb_estimator/dense_2/bias*
_output_shapes
:*
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
?
 Adam/wb_estimator/dense/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:		*1
shared_name" Adam/wb_estimator/dense/kernel/m
?
4Adam/wb_estimator/dense/kernel/m/Read/ReadVariableOpReadVariableOp Adam/wb_estimator/dense/kernel/m*
_output_shapes

:		*
dtype0
?
Adam/wb_estimator/dense/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*/
shared_name Adam/wb_estimator/dense/bias/m
?
2Adam/wb_estimator/dense/bias/m/Read/ReadVariableOpReadVariableOpAdam/wb_estimator/dense/bias/m*
_output_shapes
:	*
dtype0
?
"Adam/wb_estimator/dense_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:		*3
shared_name$"Adam/wb_estimator/dense_1/kernel/m
?
6Adam/wb_estimator/dense_1/kernel/m/Read/ReadVariableOpReadVariableOp"Adam/wb_estimator/dense_1/kernel/m*
_output_shapes

:		*
dtype0
?
 Adam/wb_estimator/dense_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*1
shared_name" Adam/wb_estimator/dense_1/bias/m
?
4Adam/wb_estimator/dense_1/bias/m/Read/ReadVariableOpReadVariableOp Adam/wb_estimator/dense_1/bias/m*
_output_shapes
:	*
dtype0
?
"Adam/wb_estimator/dense_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:	*3
shared_name$"Adam/wb_estimator/dense_2/kernel/m
?
6Adam/wb_estimator/dense_2/kernel/m/Read/ReadVariableOpReadVariableOp"Adam/wb_estimator/dense_2/kernel/m*
_output_shapes

:	*
dtype0
?
 Adam/wb_estimator/dense_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" Adam/wb_estimator/dense_2/bias/m
?
4Adam/wb_estimator/dense_2/bias/m/Read/ReadVariableOpReadVariableOp Adam/wb_estimator/dense_2/bias/m*
_output_shapes
:*
dtype0
?
 Adam/wb_estimator/dense/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:		*1
shared_name" Adam/wb_estimator/dense/kernel/v
?
4Adam/wb_estimator/dense/kernel/v/Read/ReadVariableOpReadVariableOp Adam/wb_estimator/dense/kernel/v*
_output_shapes

:		*
dtype0
?
Adam/wb_estimator/dense/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*/
shared_name Adam/wb_estimator/dense/bias/v
?
2Adam/wb_estimator/dense/bias/v/Read/ReadVariableOpReadVariableOpAdam/wb_estimator/dense/bias/v*
_output_shapes
:	*
dtype0
?
"Adam/wb_estimator/dense_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:		*3
shared_name$"Adam/wb_estimator/dense_1/kernel/v
?
6Adam/wb_estimator/dense_1/kernel/v/Read/ReadVariableOpReadVariableOp"Adam/wb_estimator/dense_1/kernel/v*
_output_shapes

:		*
dtype0
?
 Adam/wb_estimator/dense_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*1
shared_name" Adam/wb_estimator/dense_1/bias/v
?
4Adam/wb_estimator/dense_1/bias/v/Read/ReadVariableOpReadVariableOp Adam/wb_estimator/dense_1/bias/v*
_output_shapes
:	*
dtype0
?
"Adam/wb_estimator/dense_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:	*3
shared_name$"Adam/wb_estimator/dense_2/kernel/v
?
6Adam/wb_estimator/dense_2/kernel/v/Read/ReadVariableOpReadVariableOp"Adam/wb_estimator/dense_2/kernel/v*
_output_shapes

:	*
dtype0
?
 Adam/wb_estimator/dense_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" Adam/wb_estimator/dense_2/bias/v
?
4Adam/wb_estimator/dense_2/bias/v/Read/ReadVariableOpReadVariableOp Adam/wb_estimator/dense_2/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
?*
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?*
value?*B?* B?*
~
layers_
	optimizer
regularization_losses
	variables
trainable_variables
	keras_api

signatures
*
0
	1

2
3
4
5
?

beta_1

beta_2
	decay
learning_rate
iterm_m`mambmcmdvevfvgvhvivj
 
*
0
1
2
3
4
5
*
0
1
2
3
4
5
?
layer_regularization_losses
non_trainable_variables
regularization_losses
	variables
trainable_variables

layers
metrics
layer_metrics
 
R
regularization_losses
	variables
 trainable_variables
!	keras_api
h

kernel
bias
"regularization_losses
#	variables
$trainable_variables
%	keras_api
h

kernel
bias
&regularization_losses
'	variables
(trainable_variables
)	keras_api
h

kernel
bias
*regularization_losses
+	variables
,trainable_variables
-	keras_api
R
.regularization_losses
/	variables
0trainable_variables
1	keras_api
R
2regularization_losses
3	variables
4trainable_variables
5	keras_api
GE
VARIABLE_VALUEbeta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
GE
VARIABLE_VALUEbeta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
EC
VARIABLE_VALUEdecay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUElearning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEwb_estimator/dense/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUEwb_estimator/dense/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEwb_estimator/dense_1/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEwb_estimator/dense_1/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEwb_estimator/dense_2/kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEwb_estimator/dense_2/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE
 
 
*
0
	1

2
3
4
5

60
71
 
 
 
 
?
8layer_regularization_losses
9non_trainable_variables
regularization_losses
	variables
 trainable_variables

:layers
;metrics
<layer_metrics
 

0
1

0
1
?
=layer_regularization_losses
>non_trainable_variables
"regularization_losses
#	variables
$trainable_variables

?layers
@metrics
Alayer_metrics
 

0
1

0
1
?
Blayer_regularization_losses
Cnon_trainable_variables
&regularization_losses
'	variables
(trainable_variables

Dlayers
Emetrics
Flayer_metrics
 

0
1

0
1
?
Glayer_regularization_losses
Hnon_trainable_variables
*regularization_losses
+	variables
,trainable_variables

Ilayers
Jmetrics
Klayer_metrics
 
 
 
?
Llayer_regularization_losses
Mnon_trainable_variables
.regularization_losses
/	variables
0trainable_variables

Nlayers
Ometrics
Player_metrics
 
 
 
?
Qlayer_regularization_losses
Rnon_trainable_variables
2regularization_losses
3	variables
4trainable_variables

Slayers
Tmetrics
Ulayer_metrics
4
	Vtotal
	Wcount
X	variables
Y	keras_api
D
	Ztotal
	[count
\
_fn_kwargs
]	variables
^	keras_api
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

V0
W1

X	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

Z0
[1

]	variables
xv
VARIABLE_VALUE Adam/wb_estimator/dense/kernel/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/wb_estimator/dense/bias/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE"Adam/wb_estimator/dense_1/kernel/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUE Adam/wb_estimator/dense_1/bias/mBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE"Adam/wb_estimator/dense_2/kernel/mBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUE Adam/wb_estimator/dense_2/bias/mBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUE Adam/wb_estimator/dense/kernel/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/wb_estimator/dense/bias/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE"Adam/wb_estimator/dense_1/kernel/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUE Adam/wb_estimator/dense_1/bias/vBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE"Adam/wb_estimator/dense_2/kernel/vBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUE Adam/wb_estimator/dense_2/bias/vBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
serving_default_input_1Placeholder*+
_output_shapes
:?????????*
dtype0* 
shape:?????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1wb_estimator/dense/kernelwb_estimator/dense/biaswb_estimator/dense_1/kernelwb_estimator/dense_1/biaswb_estimator/dense_2/kernelwb_estimator/dense_2/bias*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? */
f*R(
&__inference_signature_wrapper_25995710
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenamebeta_1/Read/ReadVariableOpbeta_2/Read/ReadVariableOpdecay/Read/ReadVariableOp!learning_rate/Read/ReadVariableOpAdam/iter/Read/ReadVariableOp-wb_estimator/dense/kernel/Read/ReadVariableOp+wb_estimator/dense/bias/Read/ReadVariableOp/wb_estimator/dense_1/kernel/Read/ReadVariableOp-wb_estimator/dense_1/bias/Read/ReadVariableOp/wb_estimator/dense_2/kernel/Read/ReadVariableOp-wb_estimator/dense_2/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp4Adam/wb_estimator/dense/kernel/m/Read/ReadVariableOp2Adam/wb_estimator/dense/bias/m/Read/ReadVariableOp6Adam/wb_estimator/dense_1/kernel/m/Read/ReadVariableOp4Adam/wb_estimator/dense_1/bias/m/Read/ReadVariableOp6Adam/wb_estimator/dense_2/kernel/m/Read/ReadVariableOp4Adam/wb_estimator/dense_2/bias/m/Read/ReadVariableOp4Adam/wb_estimator/dense/kernel/v/Read/ReadVariableOp2Adam/wb_estimator/dense/bias/v/Read/ReadVariableOp6Adam/wb_estimator/dense_1/kernel/v/Read/ReadVariableOp4Adam/wb_estimator/dense_1/bias/v/Read/ReadVariableOp6Adam/wb_estimator/dense_2/kernel/v/Read/ReadVariableOp4Adam/wb_estimator/dense_2/bias/v/Read/ReadVariableOpConst*(
Tin!
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? **
f%R#
!__inference__traced_save_25996126
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamebeta_1beta_2decaylearning_rate	Adam/iterwb_estimator/dense/kernelwb_estimator/dense/biaswb_estimator/dense_1/kernelwb_estimator/dense_1/biaswb_estimator/dense_2/kernelwb_estimator/dense_2/biastotalcounttotal_1count_1 Adam/wb_estimator/dense/kernel/mAdam/wb_estimator/dense/bias/m"Adam/wb_estimator/dense_1/kernel/m Adam/wb_estimator/dense_1/bias/m"Adam/wb_estimator/dense_2/kernel/m Adam/wb_estimator/dense_2/bias/m Adam/wb_estimator/dense/kernel/vAdam/wb_estimator/dense/bias/v"Adam/wb_estimator/dense_1/kernel/v Adam/wb_estimator/dense_1/bias/v"Adam/wb_estimator/dense_2/kernel/v Adam/wb_estimator/dense_2/bias/v*'
Tin 
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *-
f(R&
$__inference__traced_restore_25996217??
?'
h
L__inference_adjust_output1_layer_call_and_return_conditional_losses_25995454

inputs
identity{
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice/stack_2?
strided_sliceStridedSliceinputsstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*
ellipsis_mask*
shrink_axis_mask2
strided_slice
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_1/stack?
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_1/stack_1?
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_1/stack_2?
strided_slice_1StridedSlicestrided_slice:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
ellipsis_mask*
new_axis_mask2
strided_slice_1
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_2/stack?
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_2/stack_1?
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_2/stack_2?
strided_slice_2StridedSliceinputsstrided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*
ellipsis_mask*
shrink_axis_mask2
strided_slice_2j
ones_like/ShapeShapestrided_slice_2:output:0*
T0*
_output_shapes
:2
ones_like/Shapeg
ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
ones_like/Const?
	ones_likeFillones_like/Shape:output:0ones_like/Const:output:0*
T0*#
_output_shapes
:?????????2
	ones_like
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_3/stack?
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_3/stack_1?
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_3/stack_2?
strided_slice_3StridedSliceones_like:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
ellipsis_mask*
new_axis_mask2
strided_slice_3
strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_4/stack?
strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_4/stack_1?
strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_4/stack_2?
strided_slice_4StridedSliceinputsstrided_slice_4/stack:output:0 strided_slice_4/stack_1:output:0 strided_slice_4/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*
ellipsis_mask*
shrink_axis_mask2
strided_slice_4
strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_5/stack?
strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_5/stack_1?
strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_5/stack_2?
strided_slice_5StridedSlicestrided_slice_4:output:0strided_slice_5/stack:output:0 strided_slice_5/stack_1:output:0 strided_slice_5/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
ellipsis_mask*
new_axis_mask2
strided_slice_5\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis?
concatConcatV2strided_slice_1:output:0strided_slice_3:output:0strided_slice_5:output:0concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????2
concatc
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?>
?
!__inference__traced_save_25996126
file_prefix%
!savev2_beta_1_read_readvariableop%
!savev2_beta_2_read_readvariableop$
 savev2_decay_read_readvariableop,
(savev2_learning_rate_read_readvariableop(
$savev2_adam_iter_read_readvariableop	8
4savev2_wb_estimator_dense_kernel_read_readvariableop6
2savev2_wb_estimator_dense_bias_read_readvariableop:
6savev2_wb_estimator_dense_1_kernel_read_readvariableop8
4savev2_wb_estimator_dense_1_bias_read_readvariableop:
6savev2_wb_estimator_dense_2_kernel_read_readvariableop8
4savev2_wb_estimator_dense_2_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop?
;savev2_adam_wb_estimator_dense_kernel_m_read_readvariableop=
9savev2_adam_wb_estimator_dense_bias_m_read_readvariableopA
=savev2_adam_wb_estimator_dense_1_kernel_m_read_readvariableop?
;savev2_adam_wb_estimator_dense_1_bias_m_read_readvariableopA
=savev2_adam_wb_estimator_dense_2_kernel_m_read_readvariableop?
;savev2_adam_wb_estimator_dense_2_bias_m_read_readvariableop?
;savev2_adam_wb_estimator_dense_kernel_v_read_readvariableop=
9savev2_adam_wb_estimator_dense_bias_v_read_readvariableopA
=savev2_adam_wb_estimator_dense_1_kernel_v_read_readvariableop?
;savev2_adam_wb_estimator_dense_1_bias_v_read_readvariableopA
=savev2_adam_wb_estimator_dense_2_kernel_v_read_readvariableop?
;savev2_adam_wb_estimator_dense_2_bias_v_read_readvariableop
savev2_const

identity_1??MergeV2Checkpoints?
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*K
valueBB@B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0!savev2_beta_1_read_readvariableop!savev2_beta_2_read_readvariableop savev2_decay_read_readvariableop(savev2_learning_rate_read_readvariableop$savev2_adam_iter_read_readvariableop4savev2_wb_estimator_dense_kernel_read_readvariableop2savev2_wb_estimator_dense_bias_read_readvariableop6savev2_wb_estimator_dense_1_kernel_read_readvariableop4savev2_wb_estimator_dense_1_bias_read_readvariableop6savev2_wb_estimator_dense_2_kernel_read_readvariableop4savev2_wb_estimator_dense_2_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop;savev2_adam_wb_estimator_dense_kernel_m_read_readvariableop9savev2_adam_wb_estimator_dense_bias_m_read_readvariableop=savev2_adam_wb_estimator_dense_1_kernel_m_read_readvariableop;savev2_adam_wb_estimator_dense_1_bias_m_read_readvariableop=savev2_adam_wb_estimator_dense_2_kernel_m_read_readvariableop;savev2_adam_wb_estimator_dense_2_bias_m_read_readvariableop;savev2_adam_wb_estimator_dense_kernel_v_read_readvariableop9savev2_adam_wb_estimator_dense_bias_v_read_readvariableop=savev2_adam_wb_estimator_dense_1_kernel_v_read_readvariableop;savev2_adam_wb_estimator_dense_1_bias_v_read_readvariableop=savev2_adam_wb_estimator_dense_2_kernel_v_read_readvariableop;savev2_adam_wb_estimator_dense_2_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 **
dtypes 
2	2
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*?
_input_shapes?
?: : : : : : :		:	:		:	:	:: : : : :		:	:		:	:	::		:	:		:	:	:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:		: 

_output_shapes
:	:$ 

_output_shapes

:		: 	

_output_shapes
:	:$
 

_output_shapes

:	: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:		: 

_output_shapes
:	:$ 

_output_shapes

:		: 

_output_shapes
:	:$ 

_output_shapes

:	: 

_output_shapes
::$ 

_output_shapes

:		: 

_output_shapes
:	:$ 

_output_shapes

:		: 

_output_shapes
:	:$ 

_output_shapes

:	: 

_output_shapes
::

_output_shapes
: 
?
M
1__inference_adjust_output1_layer_call_fn_25995956

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_adjust_output1_layer_call_and_return_conditional_losses_259955132
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
(__inference_dense_layer_call_fn_25995876

inputs
unknown:		
	unknown_0:	
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????	*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_dense_layer_call_and_return_conditional_losses_259953752
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????	2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????	: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????	
 
_user_specified_nameinputs
?_
?
#__inference__wrapped_model_25995349
input_1C
1wb_estimator_dense_matmul_readvariableop_resource:		@
2wb_estimator_dense_biasadd_readvariableop_resource:	E
3wb_estimator_dense_1_matmul_readvariableop_resource:		B
4wb_estimator_dense_1_biasadd_readvariableop_resource:	E
3wb_estimator_dense_2_matmul_readvariableop_resource:	B
4wb_estimator_dense_2_biasadd_readvariableop_resource:
identity??)wb_estimator/dense/BiasAdd/ReadVariableOp?(wb_estimator/dense/MatMul/ReadVariableOp?+wb_estimator/dense_1/BiasAdd/ReadVariableOp?*wb_estimator/dense_1/MatMul/ReadVariableOp?+wb_estimator/dense_2/BiasAdd/ReadVariableOp?*wb_estimator/dense_2/MatMul/ReadVariableOp?
wb_estimator/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????	   2
wb_estimator/flatten/Const?
wb_estimator/flatten/ReshapeReshapeinput_1#wb_estimator/flatten/Const:output:0*
T0*'
_output_shapes
:?????????	2
wb_estimator/flatten/Reshape?
(wb_estimator/dense/MatMul/ReadVariableOpReadVariableOp1wb_estimator_dense_matmul_readvariableop_resource*
_output_shapes

:		*
dtype02*
(wb_estimator/dense/MatMul/ReadVariableOp?
wb_estimator/dense/MatMulMatMul%wb_estimator/flatten/Reshape:output:00wb_estimator/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????	2
wb_estimator/dense/MatMul?
)wb_estimator/dense/BiasAdd/ReadVariableOpReadVariableOp2wb_estimator_dense_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype02+
)wb_estimator/dense/BiasAdd/ReadVariableOp?
wb_estimator/dense/BiasAddBiasAdd#wb_estimator/dense/MatMul:product:01wb_estimator/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????	2
wb_estimator/dense/BiasAdd?
wb_estimator/dense/ReluRelu#wb_estimator/dense/BiasAdd:output:0*
T0*'
_output_shapes
:?????????	2
wb_estimator/dense/Relu?
*wb_estimator/dense_1/MatMul/ReadVariableOpReadVariableOp3wb_estimator_dense_1_matmul_readvariableop_resource*
_output_shapes

:		*
dtype02,
*wb_estimator/dense_1/MatMul/ReadVariableOp?
wb_estimator/dense_1/MatMulMatMul%wb_estimator/dense/Relu:activations:02wb_estimator/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????	2
wb_estimator/dense_1/MatMul?
+wb_estimator/dense_1/BiasAdd/ReadVariableOpReadVariableOp4wb_estimator_dense_1_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype02-
+wb_estimator/dense_1/BiasAdd/ReadVariableOp?
wb_estimator/dense_1/BiasAddBiasAdd%wb_estimator/dense_1/MatMul:product:03wb_estimator/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????	2
wb_estimator/dense_1/BiasAdd?
wb_estimator/dense_1/ReluRelu%wb_estimator/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????	2
wb_estimator/dense_1/Relu?
*wb_estimator/dense_2/MatMul/ReadVariableOpReadVariableOp3wb_estimator_dense_2_matmul_readvariableop_resource*
_output_shapes

:	*
dtype02,
*wb_estimator/dense_2/MatMul/ReadVariableOp?
wb_estimator/dense_2/MatMulMatMul'wb_estimator/dense_1/Relu:activations:02wb_estimator/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
wb_estimator/dense_2/MatMul?
+wb_estimator/dense_2/BiasAdd/ReadVariableOpReadVariableOp4wb_estimator_dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02-
+wb_estimator/dense_2/BiasAdd/ReadVariableOp?
wb_estimator/dense_2/BiasAddBiasAdd%wb_estimator/dense_2/MatMul:product:03wb_estimator/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
wb_estimator/dense_2/BiasAdd?
wb_estimator/exp/ExpExp%wb_estimator/dense_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
wb_estimator/exp/Exp?
/wb_estimator/adjust_output1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        21
/wb_estimator/adjust_output1/strided_slice/stack?
1wb_estimator/adjust_output1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       23
1wb_estimator/adjust_output1/strided_slice/stack_1?
1wb_estimator/adjust_output1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      23
1wb_estimator/adjust_output1/strided_slice/stack_2?
)wb_estimator/adjust_output1/strided_sliceStridedSlicewb_estimator/exp/Exp:y:08wb_estimator/adjust_output1/strided_slice/stack:output:0:wb_estimator/adjust_output1/strided_slice/stack_1:output:0:wb_estimator/adjust_output1/strided_slice/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*
ellipsis_mask*
shrink_axis_mask2+
)wb_estimator/adjust_output1/strided_slice?
1wb_estimator/adjust_output1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        23
1wb_estimator/adjust_output1/strided_slice_1/stack?
3wb_estimator/adjust_output1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        25
3wb_estimator/adjust_output1/strided_slice_1/stack_1?
3wb_estimator/adjust_output1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      25
3wb_estimator/adjust_output1/strided_slice_1/stack_2?
+wb_estimator/adjust_output1/strided_slice_1StridedSlice2wb_estimator/adjust_output1/strided_slice:output:0:wb_estimator/adjust_output1/strided_slice_1/stack:output:0<wb_estimator/adjust_output1/strided_slice_1/stack_1:output:0<wb_estimator/adjust_output1/strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
ellipsis_mask*
new_axis_mask2-
+wb_estimator/adjust_output1/strided_slice_1?
1wb_estimator/adjust_output1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        23
1wb_estimator/adjust_output1/strided_slice_2/stack?
3wb_estimator/adjust_output1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       25
3wb_estimator/adjust_output1/strided_slice_2/stack_1?
3wb_estimator/adjust_output1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      25
3wb_estimator/adjust_output1/strided_slice_2/stack_2?
+wb_estimator/adjust_output1/strided_slice_2StridedSlicewb_estimator/exp/Exp:y:0:wb_estimator/adjust_output1/strided_slice_2/stack:output:0<wb_estimator/adjust_output1/strided_slice_2/stack_1:output:0<wb_estimator/adjust_output1/strided_slice_2/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*
ellipsis_mask*
shrink_axis_mask2-
+wb_estimator/adjust_output1/strided_slice_2?
+wb_estimator/adjust_output1/ones_like/ShapeShape4wb_estimator/adjust_output1/strided_slice_2:output:0*
T0*
_output_shapes
:2-
+wb_estimator/adjust_output1/ones_like/Shape?
+wb_estimator/adjust_output1/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2-
+wb_estimator/adjust_output1/ones_like/Const?
%wb_estimator/adjust_output1/ones_likeFill4wb_estimator/adjust_output1/ones_like/Shape:output:04wb_estimator/adjust_output1/ones_like/Const:output:0*
T0*#
_output_shapes
:?????????2'
%wb_estimator/adjust_output1/ones_like?
1wb_estimator/adjust_output1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"        23
1wb_estimator/adjust_output1/strided_slice_3/stack?
3wb_estimator/adjust_output1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        25
3wb_estimator/adjust_output1/strided_slice_3/stack_1?
3wb_estimator/adjust_output1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      25
3wb_estimator/adjust_output1/strided_slice_3/stack_2?
+wb_estimator/adjust_output1/strided_slice_3StridedSlice.wb_estimator/adjust_output1/ones_like:output:0:wb_estimator/adjust_output1/strided_slice_3/stack:output:0<wb_estimator/adjust_output1/strided_slice_3/stack_1:output:0<wb_estimator/adjust_output1/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
ellipsis_mask*
new_axis_mask2-
+wb_estimator/adjust_output1/strided_slice_3?
1wb_estimator/adjust_output1/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB"       23
1wb_estimator/adjust_output1/strided_slice_4/stack?
3wb_estimator/adjust_output1/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       25
3wb_estimator/adjust_output1/strided_slice_4/stack_1?
3wb_estimator/adjust_output1/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      25
3wb_estimator/adjust_output1/strided_slice_4/stack_2?
+wb_estimator/adjust_output1/strided_slice_4StridedSlicewb_estimator/exp/Exp:y:0:wb_estimator/adjust_output1/strided_slice_4/stack:output:0<wb_estimator/adjust_output1/strided_slice_4/stack_1:output:0<wb_estimator/adjust_output1/strided_slice_4/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*
ellipsis_mask*
shrink_axis_mask2-
+wb_estimator/adjust_output1/strided_slice_4?
1wb_estimator/adjust_output1/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB"        23
1wb_estimator/adjust_output1/strided_slice_5/stack?
3wb_estimator/adjust_output1/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        25
3wb_estimator/adjust_output1/strided_slice_5/stack_1?
3wb_estimator/adjust_output1/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      25
3wb_estimator/adjust_output1/strided_slice_5/stack_2?
+wb_estimator/adjust_output1/strided_slice_5StridedSlice4wb_estimator/adjust_output1/strided_slice_4:output:0:wb_estimator/adjust_output1/strided_slice_5/stack:output:0<wb_estimator/adjust_output1/strided_slice_5/stack_1:output:0<wb_estimator/adjust_output1/strided_slice_5/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
ellipsis_mask*
new_axis_mask2-
+wb_estimator/adjust_output1/strided_slice_5?
'wb_estimator/adjust_output1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2)
'wb_estimator/adjust_output1/concat/axis?
"wb_estimator/adjust_output1/concatConcatV24wb_estimator/adjust_output1/strided_slice_1:output:04wb_estimator/adjust_output1/strided_slice_3:output:04wb_estimator/adjust_output1/strided_slice_5:output:00wb_estimator/adjust_output1/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????2$
"wb_estimator/adjust_output1/concat?
IdentityIdentity+wb_estimator/adjust_output1/concat:output:0*^wb_estimator/dense/BiasAdd/ReadVariableOp)^wb_estimator/dense/MatMul/ReadVariableOp,^wb_estimator/dense_1/BiasAdd/ReadVariableOp+^wb_estimator/dense_1/MatMul/ReadVariableOp,^wb_estimator/dense_2/BiasAdd/ReadVariableOp+^wb_estimator/dense_2/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : : : 2V
)wb_estimator/dense/BiasAdd/ReadVariableOp)wb_estimator/dense/BiasAdd/ReadVariableOp2T
(wb_estimator/dense/MatMul/ReadVariableOp(wb_estimator/dense/MatMul/ReadVariableOp2Z
+wb_estimator/dense_1/BiasAdd/ReadVariableOp+wb_estimator/dense_1/BiasAdd/ReadVariableOp2X
*wb_estimator/dense_1/MatMul/ReadVariableOp*wb_estimator/dense_1/MatMul/ReadVariableOp2Z
+wb_estimator/dense_2/BiasAdd/ReadVariableOp+wb_estimator/dense_2/BiasAdd/ReadVariableOp2X
*wb_estimator/dense_2/MatMul/ReadVariableOp*wb_estimator/dense_2/MatMul/ReadVariableOp:T P
+
_output_shapes
:?????????
!
_user_specified_name	input_1
?M
?
J__inference_wb_estimator_layer_call_and_return_conditional_losses_25995800
x6
$dense_matmul_readvariableop_resource:		3
%dense_biasadd_readvariableop_resource:	8
&dense_1_matmul_readvariableop_resource:		5
'dense_1_biasadd_readvariableop_resource:	8
&dense_2_matmul_readvariableop_resource:	5
'dense_2_biasadd_readvariableop_resource:
identity??dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp?dense_2/BiasAdd/ReadVariableOp?dense_2/MatMul/ReadVariableOpo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????	   2
flatten/Constz
flatten/ReshapeReshapexflatten/Const:output:0*
T0*'
_output_shapes
:?????????	2
flatten/Reshape?
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

:		*
dtype02
dense/MatMul/ReadVariableOp?
dense/MatMulMatMulflatten/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????	2
dense/MatMul?
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype02
dense/BiasAdd/ReadVariableOp?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????	2
dense/BiasAddj

dense/ReluReludense/BiasAdd:output:0*
T0*'
_output_shapes
:?????????	2

dense/Relu?
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:		*
dtype02
dense_1/MatMul/ReadVariableOp?
dense_1/MatMulMatMuldense/Relu:activations:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????	2
dense_1/MatMul?
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype02 
dense_1/BiasAdd/ReadVariableOp?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????	2
dense_1/BiasAddp
dense_1/ReluReludense_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????	2
dense_1/Relu?
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes

:	*
dtype02
dense_2/MatMul/ReadVariableOp?
dense_2/MatMulMatMuldense_1/Relu:activations:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_2/MatMul?
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_2/BiasAdd/ReadVariableOp?
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_2/BiasAdde
exp/ExpExpdense_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2	
exp/Exp?
"adjust_output1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2$
"adjust_output1/strided_slice/stack?
$adjust_output1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2&
$adjust_output1/strided_slice/stack_1?
$adjust_output1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2&
$adjust_output1/strided_slice/stack_2?
adjust_output1/strided_sliceStridedSliceexp/Exp:y:0+adjust_output1/strided_slice/stack:output:0-adjust_output1/strided_slice/stack_1:output:0-adjust_output1/strided_slice/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*
ellipsis_mask*
shrink_axis_mask2
adjust_output1/strided_slice?
$adjust_output1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2&
$adjust_output1/strided_slice_1/stack?
&adjust_output1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2(
&adjust_output1/strided_slice_1/stack_1?
&adjust_output1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&adjust_output1/strided_slice_1/stack_2?
adjust_output1/strided_slice_1StridedSlice%adjust_output1/strided_slice:output:0-adjust_output1/strided_slice_1/stack:output:0/adjust_output1/strided_slice_1/stack_1:output:0/adjust_output1/strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
ellipsis_mask*
new_axis_mask2 
adjust_output1/strided_slice_1?
$adjust_output1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        2&
$adjust_output1/strided_slice_2/stack?
&adjust_output1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2(
&adjust_output1/strided_slice_2/stack_1?
&adjust_output1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&adjust_output1/strided_slice_2/stack_2?
adjust_output1/strided_slice_2StridedSliceexp/Exp:y:0-adjust_output1/strided_slice_2/stack:output:0/adjust_output1/strided_slice_2/stack_1:output:0/adjust_output1/strided_slice_2/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*
ellipsis_mask*
shrink_axis_mask2 
adjust_output1/strided_slice_2?
adjust_output1/ones_like/ShapeShape'adjust_output1/strided_slice_2:output:0*
T0*
_output_shapes
:2 
adjust_output1/ones_like/Shape?
adjust_output1/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2 
adjust_output1/ones_like/Const?
adjust_output1/ones_likeFill'adjust_output1/ones_like/Shape:output:0'adjust_output1/ones_like/Const:output:0*
T0*#
_output_shapes
:?????????2
adjust_output1/ones_like?
$adjust_output1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"        2&
$adjust_output1/strided_slice_3/stack?
&adjust_output1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2(
&adjust_output1/strided_slice_3/stack_1?
&adjust_output1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&adjust_output1/strided_slice_3/stack_2?
adjust_output1/strided_slice_3StridedSlice!adjust_output1/ones_like:output:0-adjust_output1/strided_slice_3/stack:output:0/adjust_output1/strided_slice_3/stack_1:output:0/adjust_output1/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
ellipsis_mask*
new_axis_mask2 
adjust_output1/strided_slice_3?
$adjust_output1/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB"       2&
$adjust_output1/strided_slice_4/stack?
&adjust_output1/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2(
&adjust_output1/strided_slice_4/stack_1?
&adjust_output1/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&adjust_output1/strided_slice_4/stack_2?
adjust_output1/strided_slice_4StridedSliceexp/Exp:y:0-adjust_output1/strided_slice_4/stack:output:0/adjust_output1/strided_slice_4/stack_1:output:0/adjust_output1/strided_slice_4/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*
ellipsis_mask*
shrink_axis_mask2 
adjust_output1/strided_slice_4?
$adjust_output1/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB"        2&
$adjust_output1/strided_slice_5/stack?
&adjust_output1/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2(
&adjust_output1/strided_slice_5/stack_1?
&adjust_output1/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&adjust_output1/strided_slice_5/stack_2?
adjust_output1/strided_slice_5StridedSlice'adjust_output1/strided_slice_4:output:0-adjust_output1/strided_slice_5/stack:output:0/adjust_output1/strided_slice_5/stack_1:output:0/adjust_output1/strided_slice_5/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
ellipsis_mask*
new_axis_mask2 
adjust_output1/strided_slice_5z
adjust_output1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
adjust_output1/concat/axis?
adjust_output1/concatConcatV2'adjust_output1/strided_slice_1:output:0'adjust_output1/strided_slice_3:output:0'adjust_output1/strided_slice_5:output:0#adjust_output1/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????2
adjust_output1/concat?
IdentityIdentityadjust_output1/concat:output:0^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp:N J
+
_output_shapes
:?????????

_user_specified_namex
?
M
1__inference_adjust_output1_layer_call_fn_25995951

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_adjust_output1_layer_call_and_return_conditional_losses_259954542
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
*__inference_dense_2_layer_call_fn_25995916

inputs
unknown:	
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dense_2_layer_call_and_return_conditional_losses_259954082
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????	: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????	
 
_user_specified_nameinputs
?

?
E__inference_dense_1_layer_call_and_return_conditional_losses_25995392

inputs0
matmul_readvariableop_resource:		-
biasadd_readvariableop_resource:	
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:		*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????	2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:	*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????	2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????	2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????	2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????	
 
_user_specified_nameinputs
?
?
/__inference_wb_estimator_layer_call_fn_25995641
input_1
unknown:		
	unknown_0:	
	unknown_1:		
	unknown_2:	
	unknown_3:	
	unknown_4:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_wb_estimator_layer_call_and_return_conditional_losses_259956092
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
+
_output_shapes
:?????????
!
_user_specified_name	input_1
?
?
J__inference_wb_estimator_layer_call_and_return_conditional_losses_25995457
x 
dense_25995376:		
dense_25995378:	"
dense_1_25995393:		
dense_1_25995395:	"
dense_2_25995409:	
dense_2_25995411:
identity??dense/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?dense_2/StatefulPartitionedCall?
flatten/PartitionedCallPartitionedCallx*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_flatten_layer_call_and_return_conditional_losses_259953622
flatten/PartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_25995376dense_25995378*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????	*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_dense_layer_call_and_return_conditional_losses_259953752
dense/StatefulPartitionedCall?
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_25995393dense_1_25995395*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????	*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dense_1_layer_call_and_return_conditional_losses_259953922!
dense_1/StatefulPartitionedCall?
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_25995409dense_2_25995411*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dense_2_layer_call_and_return_conditional_losses_259954082!
dense_2/StatefulPartitionedCall?
exp/PartitionedCallPartitionedCall(dense_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_exp_layer_call_and_return_conditional_losses_259954192
exp/PartitionedCall?
adjust_output1/PartitionedCallPartitionedCallexp/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_adjust_output1_layer_call_and_return_conditional_losses_259954542 
adjust_output1/PartitionedCall?
IdentityIdentity'adjust_output1/PartitionedCall:output:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall:N J
+
_output_shapes
:?????????

_user_specified_namex
?
?
J__inference_wb_estimator_layer_call_and_return_conditional_losses_25995663
input_1 
dense_25995645:		
dense_25995647:	"
dense_1_25995650:		
dense_1_25995652:	"
dense_2_25995655:	
dense_2_25995657:
identity??dense/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?dense_2/StatefulPartitionedCall?
flatten/PartitionedCallPartitionedCallinput_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_flatten_layer_call_and_return_conditional_losses_259953622
flatten/PartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_25995645dense_25995647*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????	*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_dense_layer_call_and_return_conditional_losses_259953752
dense/StatefulPartitionedCall?
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_25995650dense_1_25995652*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????	*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dense_1_layer_call_and_return_conditional_losses_259953922!
dense_1/StatefulPartitionedCall?
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_25995655dense_2_25995657*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dense_2_layer_call_and_return_conditional_losses_259954082!
dense_2/StatefulPartitionedCall?
exp/PartitionedCallPartitionedCall(dense_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_exp_layer_call_and_return_conditional_losses_259954192
exp/PartitionedCall?
adjust_output1/PartitionedCallPartitionedCallexp/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_adjust_output1_layer_call_and_return_conditional_losses_259954542 
adjust_output1/PartitionedCall?
IdentityIdentity'adjust_output1/PartitionedCall:output:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall:T P
+
_output_shapes
:?????????
!
_user_specified_name	input_1
?'
h
L__inference_adjust_output1_layer_call_and_return_conditional_losses_25995513

inputs
identity{
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice/stack_2?
strided_sliceStridedSliceinputsstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*
ellipsis_mask*
shrink_axis_mask2
strided_slice
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_1/stack?
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_1/stack_1?
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_1/stack_2?
strided_slice_1StridedSlicestrided_slice:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
ellipsis_mask*
new_axis_mask2
strided_slice_1
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_2/stack?
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_2/stack_1?
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_2/stack_2?
strided_slice_2StridedSliceinputsstrided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*
ellipsis_mask*
shrink_axis_mask2
strided_slice_2j
ones_like/ShapeShapestrided_slice_2:output:0*
T0*
_output_shapes
:2
ones_like/Shapeg
ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
ones_like/Const?
	ones_likeFillones_like/Shape:output:0ones_like/Const:output:0*
T0*#
_output_shapes
:?????????2
	ones_like
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_3/stack?
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_3/stack_1?
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_3/stack_2?
strided_slice_3StridedSliceones_like:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
ellipsis_mask*
new_axis_mask2
strided_slice_3
strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_4/stack?
strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_4/stack_1?
strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_4/stack_2?
strided_slice_4StridedSliceinputsstrided_slice_4/stack:output:0 strided_slice_4/stack_1:output:0 strided_slice_4/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*
ellipsis_mask*
shrink_axis_mask2
strided_slice_4
strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_5/stack?
strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_5/stack_1?
strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_5/stack_2?
strided_slice_5StridedSlicestrided_slice_4:output:0strided_slice_5/stack:output:0 strided_slice_5/stack_1:output:0 strided_slice_5/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
ellipsis_mask*
new_axis_mask2
strided_slice_5\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis?
concatConcatV2strided_slice_1:output:0strided_slice_3:output:0strided_slice_5:output:0concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????2
concatc
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
C__inference_dense_layer_call_and_return_conditional_losses_25995887

inputs0
matmul_readvariableop_resource:		-
biasadd_readvariableop_resource:	
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:		*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????	2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:	*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????	2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????	2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????	2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????	
 
_user_specified_nameinputs
?
B
&__inference_exp_layer_call_fn_25995931

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_exp_layer_call_and_return_conditional_losses_259954192
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
J__inference_wb_estimator_layer_call_and_return_conditional_losses_25995685
input_1 
dense_25995667:		
dense_25995669:	"
dense_1_25995672:		
dense_1_25995674:	"
dense_2_25995677:	
dense_2_25995679:
identity??dense/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?dense_2/StatefulPartitionedCall?
flatten/PartitionedCallPartitionedCallinput_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_flatten_layer_call_and_return_conditional_losses_259953622
flatten/PartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_25995667dense_25995669*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????	*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_dense_layer_call_and_return_conditional_losses_259953752
dense/StatefulPartitionedCall?
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_25995672dense_1_25995674*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????	*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dense_1_layer_call_and_return_conditional_losses_259953922!
dense_1/StatefulPartitionedCall?
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_25995677dense_2_25995679*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dense_2_layer_call_and_return_conditional_losses_259954082!
dense_2/StatefulPartitionedCall?
exp/PartitionedCallPartitionedCall(dense_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_exp_layer_call_and_return_conditional_losses_259955292
exp/PartitionedCall?
adjust_output1/PartitionedCallPartitionedCallexp/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_adjust_output1_layer_call_and_return_conditional_losses_259955132 
adjust_output1/PartitionedCall?
IdentityIdentity'adjust_output1/PartitionedCall:output:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall:T P
+
_output_shapes
:?????????
!
_user_specified_name	input_1
?
?
/__inference_wb_estimator_layer_call_fn_25995744
x
unknown:		
	unknown_0:	
	unknown_1:		
	unknown_2:	
	unknown_3:	
	unknown_4:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_wb_estimator_layer_call_and_return_conditional_losses_259956092
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:N J
+
_output_shapes
:?????????

_user_specified_namex
?
?
J__inference_wb_estimator_layer_call_and_return_conditional_losses_25995609
x 
dense_25995591:		
dense_25995593:	"
dense_1_25995596:		
dense_1_25995598:	"
dense_2_25995601:	
dense_2_25995603:
identity??dense/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?dense_2/StatefulPartitionedCall?
flatten/PartitionedCallPartitionedCallx*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_flatten_layer_call_and_return_conditional_losses_259953622
flatten/PartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_25995591dense_25995593*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????	*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_dense_layer_call_and_return_conditional_losses_259953752
dense/StatefulPartitionedCall?
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_25995596dense_1_25995598*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????	*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dense_1_layer_call_and_return_conditional_losses_259953922!
dense_1/StatefulPartitionedCall?
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_25995601dense_2_25995603*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dense_2_layer_call_and_return_conditional_losses_259954082!
dense_2/StatefulPartitionedCall?
exp/PartitionedCallPartitionedCall(dense_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_exp_layer_call_and_return_conditional_losses_259955292
exp/PartitionedCall?
adjust_output1/PartitionedCallPartitionedCallexp/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *U
fPRN
L__inference_adjust_output1_layer_call_and_return_conditional_losses_259955132 
adjust_output1/PartitionedCall?
IdentityIdentity'adjust_output1/PartitionedCall:output:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall:N J
+
_output_shapes
:?????????

_user_specified_namex
?
?
/__inference_wb_estimator_layer_call_fn_25995727
x
unknown:		
	unknown_0:	
	unknown_1:		
	unknown_2:	
	unknown_3:	
	unknown_4:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallxunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_wb_estimator_layer_call_and_return_conditional_losses_259954572
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:N J
+
_output_shapes
:?????????

_user_specified_namex
?
?
&__inference_signature_wrapper_25995710
input_1
unknown:		
	unknown_0:	
	unknown_1:		
	unknown_2:	
	unknown_3:	
	unknown_4:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *,
f'R%
#__inference__wrapped_model_259953492
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
+
_output_shapes
:?????????
!
_user_specified_name	input_1
?
?
*__inference_dense_1_layer_call_fn_25995896

inputs
unknown:		
	unknown_0:	
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????	*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dense_1_layer_call_and_return_conditional_losses_259953922
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????	2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????	: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????	
 
_user_specified_nameinputs
?

?
E__inference_dense_1_layer_call_and_return_conditional_losses_25995907

inputs0
matmul_readvariableop_resource:		-
biasadd_readvariableop_resource:	
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:		*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????	2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:	*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????	2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????	2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????	2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????	
 
_user_specified_nameinputs
?'
h
L__inference_adjust_output1_layer_call_and_return_conditional_losses_25995989

inputs
identity{
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice/stack_2?
strided_sliceStridedSliceinputsstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*
ellipsis_mask*
shrink_axis_mask2
strided_slice
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_1/stack?
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_1/stack_1?
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_1/stack_2?
strided_slice_1StridedSlicestrided_slice:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
ellipsis_mask*
new_axis_mask2
strided_slice_1
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_2/stack?
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_2/stack_1?
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_2/stack_2?
strided_slice_2StridedSliceinputsstrided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*
ellipsis_mask*
shrink_axis_mask2
strided_slice_2j
ones_like/ShapeShapestrided_slice_2:output:0*
T0*
_output_shapes
:2
ones_like/Shapeg
ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
ones_like/Const?
	ones_likeFillones_like/Shape:output:0ones_like/Const:output:0*
T0*#
_output_shapes
:?????????2
	ones_like
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_3/stack?
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_3/stack_1?
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_3/stack_2?
strided_slice_3StridedSliceones_like:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
ellipsis_mask*
new_axis_mask2
strided_slice_3
strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_4/stack?
strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_4/stack_1?
strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_4/stack_2?
strided_slice_4StridedSliceinputsstrided_slice_4/stack:output:0 strided_slice_4/stack_1:output:0 strided_slice_4/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*
ellipsis_mask*
shrink_axis_mask2
strided_slice_4
strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_5/stack?
strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_5/stack_1?
strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_5/stack_2?
strided_slice_5StridedSlicestrided_slice_4:output:0strided_slice_5/stack:output:0 strided_slice_5/stack_1:output:0 strided_slice_5/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
ellipsis_mask*
new_axis_mask2
strided_slice_5\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis?
concatConcatV2strided_slice_1:output:0strided_slice_3:output:0strided_slice_5:output:0concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????2
concatc
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
C__inference_dense_layer_call_and_return_conditional_losses_25995375

inputs0
matmul_readvariableop_resource:		-
biasadd_readvariableop_resource:	
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:		*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????	2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:	*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????	2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????	2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????	2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????	
 
_user_specified_nameinputs
?
]
A__inference_exp_layer_call_and_return_conditional_losses_25995941

inputs
identityK
ExpExpinputs*
T0*'
_output_shapes
:?????????2
Exp[
IdentityIdentityExp:y:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?u
?
$__inference__traced_restore_25996217
file_prefix!
assignvariableop_beta_1: #
assignvariableop_1_beta_2: "
assignvariableop_2_decay: *
 assignvariableop_3_learning_rate: &
assignvariableop_4_adam_iter:	 >
,assignvariableop_5_wb_estimator_dense_kernel:		8
*assignvariableop_6_wb_estimator_dense_bias:	@
.assignvariableop_7_wb_estimator_dense_1_kernel:		:
,assignvariableop_8_wb_estimator_dense_1_bias:	@
.assignvariableop_9_wb_estimator_dense_2_kernel:	;
-assignvariableop_10_wb_estimator_dense_2_bias:#
assignvariableop_11_total: #
assignvariableop_12_count: %
assignvariableop_13_total_1: %
assignvariableop_14_count_1: F
4assignvariableop_15_adam_wb_estimator_dense_kernel_m:		@
2assignvariableop_16_adam_wb_estimator_dense_bias_m:	H
6assignvariableop_17_adam_wb_estimator_dense_1_kernel_m:		B
4assignvariableop_18_adam_wb_estimator_dense_1_bias_m:	H
6assignvariableop_19_adam_wb_estimator_dense_2_kernel_m:	B
4assignvariableop_20_adam_wb_estimator_dense_2_bias_m:F
4assignvariableop_21_adam_wb_estimator_dense_kernel_v:		@
2assignvariableop_22_adam_wb_estimator_dense_bias_v:	H
6assignvariableop_23_adam_wb_estimator_dense_1_kernel_v:		B
4assignvariableop_24_adam_wb_estimator_dense_1_bias_v:	H
6assignvariableop_25_adam_wb_estimator_dense_2_kernel_v:	B
4assignvariableop_26_adam_wb_estimator_dense_2_bias_v:
identity_28??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_3?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*K
valueBB@B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapesr
p::::::::::::::::::::::::::::**
dtypes 
2	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOpassignvariableop_beta_1Identity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOpassignvariableop_1_beta_2Identity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOpassignvariableop_2_decayIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp assignvariableop_3_learning_rateIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOpassignvariableop_4_adam_iterIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp,assignvariableop_5_wb_estimator_dense_kernelIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp*assignvariableop_6_wb_estimator_dense_biasIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOp.assignvariableop_7_wb_estimator_dense_1_kernelIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp,assignvariableop_8_wb_estimator_dense_1_biasIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp.assignvariableop_9_wb_estimator_dense_2_kernelIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp-assignvariableop_10_wb_estimator_dense_2_biasIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOpassignvariableop_11_totalIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOpassignvariableop_12_countIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOpassignvariableop_13_total_1Identity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOpassignvariableop_14_count_1Identity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOp4assignvariableop_15_adam_wb_estimator_dense_kernel_mIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOp2assignvariableop_16_adam_wb_estimator_dense_bias_mIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOp6assignvariableop_17_adam_wb_estimator_dense_1_kernel_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOp4assignvariableop_18_adam_wb_estimator_dense_1_bias_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOp6assignvariableop_19_adam_wb_estimator_dense_2_kernel_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOp4assignvariableop_20_adam_wb_estimator_dense_2_bias_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOp4assignvariableop_21_adam_wb_estimator_dense_kernel_vIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOp2assignvariableop_22_adam_wb_estimator_dense_bias_vIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOp6assignvariableop_23_adam_wb_estimator_dense_1_kernel_vIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOp4assignvariableop_24_adam_wb_estimator_dense_1_bias_vIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOp6assignvariableop_25_adam_wb_estimator_dense_2_kernel_vIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOp4assignvariableop_26_adam_wb_estimator_dense_2_bias_vIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_269
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_27Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_27?
Identity_28IdentityIdentity_27:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_28"#
identity_28Identity_28:output:0*K
_input_shapes:
8: : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?
]
A__inference_exp_layer_call_and_return_conditional_losses_25995529

inputs
identityK
ExpExpinputs*
T0*'
_output_shapes
:?????????2
Exp[
IdentityIdentityExp:y:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
/__inference_wb_estimator_layer_call_fn_25995472
input_1
unknown:		
	unknown_0:	
	unknown_1:		
	unknown_2:	
	unknown_3:	
	unknown_4:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_wb_estimator_layer_call_and_return_conditional_losses_259954572
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
+
_output_shapes
:?????????
!
_user_specified_name	input_1
?
]
A__inference_exp_layer_call_and_return_conditional_losses_25995419

inputs
identityK
ExpExpinputs*
T0*'
_output_shapes
:?????????2
Exp[
IdentityIdentityExp:y:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
B
&__inference_exp_layer_call_fn_25995936

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_exp_layer_call_and_return_conditional_losses_259955292
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?'
h
L__inference_adjust_output1_layer_call_and_return_conditional_losses_25996022

inputs
identity{
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice/stack_2?
strided_sliceStridedSliceinputsstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*
ellipsis_mask*
shrink_axis_mask2
strided_slice
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_1/stack?
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_1/stack_1?
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_1/stack_2?
strided_slice_1StridedSlicestrided_slice:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
ellipsis_mask*
new_axis_mask2
strided_slice_1
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_2/stack?
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_2/stack_1?
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_2/stack_2?
strided_slice_2StridedSliceinputsstrided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*
ellipsis_mask*
shrink_axis_mask2
strided_slice_2j
ones_like/ShapeShapestrided_slice_2:output:0*
T0*
_output_shapes
:2
ones_like/Shapeg
ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
ones_like/Const?
	ones_likeFillones_like/Shape:output:0ones_like/Const:output:0*
T0*#
_output_shapes
:?????????2
	ones_like
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_3/stack?
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_3/stack_1?
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_3/stack_2?
strided_slice_3StridedSliceones_like:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
ellipsis_mask*
new_axis_mask2
strided_slice_3
strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_4/stack?
strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2
strided_slice_4/stack_1?
strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_4/stack_2?
strided_slice_4StridedSliceinputsstrided_slice_4/stack:output:0 strided_slice_4/stack_1:output:0 strided_slice_4/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*
ellipsis_mask*
shrink_axis_mask2
strided_slice_4
strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_5/stack?
strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_5/stack_1?
strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_5/stack_2?
strided_slice_5StridedSlicestrided_slice_4:output:0strided_slice_5/stack:output:0 strided_slice_5/stack_1:output:0 strided_slice_5/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
ellipsis_mask*
new_axis_mask2
strided_slice_5\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis?
concatConcatV2strided_slice_1:output:0strided_slice_3:output:0strided_slice_5:output:0concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????2
concatc
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
]
A__inference_exp_layer_call_and_return_conditional_losses_25995946

inputs
identityK
ExpExpinputs*
T0*'
_output_shapes
:?????????2
Exp[
IdentityIdentityExp:y:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
a
E__inference_flatten_layer_call_and_return_conditional_losses_25995362

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????	   2
Constg
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:?????????	2	
Reshaped
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:?????????	2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?M
?
J__inference_wb_estimator_layer_call_and_return_conditional_losses_25995856
x6
$dense_matmul_readvariableop_resource:		3
%dense_biasadd_readvariableop_resource:	8
&dense_1_matmul_readvariableop_resource:		5
'dense_1_biasadd_readvariableop_resource:	8
&dense_2_matmul_readvariableop_resource:	5
'dense_2_biasadd_readvariableop_resource:
identity??dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp?dense_2/BiasAdd/ReadVariableOp?dense_2/MatMul/ReadVariableOpo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????	   2
flatten/Constz
flatten/ReshapeReshapexflatten/Const:output:0*
T0*'
_output_shapes
:?????????	2
flatten/Reshape?
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

:		*
dtype02
dense/MatMul/ReadVariableOp?
dense/MatMulMatMulflatten/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????	2
dense/MatMul?
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype02
dense/BiasAdd/ReadVariableOp?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????	2
dense/BiasAddj

dense/ReluReludense/BiasAdd:output:0*
T0*'
_output_shapes
:?????????	2

dense/Relu?
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:		*
dtype02
dense_1/MatMul/ReadVariableOp?
dense_1/MatMulMatMuldense/Relu:activations:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????	2
dense_1/MatMul?
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype02 
dense_1/BiasAdd/ReadVariableOp?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????	2
dense_1/BiasAddp
dense_1/ReluReludense_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????	2
dense_1/Relu?
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes

:	*
dtype02
dense_2/MatMul/ReadVariableOp?
dense_2/MatMulMatMuldense_1/Relu:activations:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_2/MatMul?
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_2/BiasAdd/ReadVariableOp?
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_2/BiasAdde
exp/ExpExpdense_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2	
exp/Exp?
"adjust_output1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2$
"adjust_output1/strided_slice/stack?
$adjust_output1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2&
$adjust_output1/strided_slice/stack_1?
$adjust_output1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2&
$adjust_output1/strided_slice/stack_2?
adjust_output1/strided_sliceStridedSliceexp/Exp:y:0+adjust_output1/strided_slice/stack:output:0-adjust_output1/strided_slice/stack_1:output:0-adjust_output1/strided_slice/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*
ellipsis_mask*
shrink_axis_mask2
adjust_output1/strided_slice?
$adjust_output1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2&
$adjust_output1/strided_slice_1/stack?
&adjust_output1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2(
&adjust_output1/strided_slice_1/stack_1?
&adjust_output1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&adjust_output1/strided_slice_1/stack_2?
adjust_output1/strided_slice_1StridedSlice%adjust_output1/strided_slice:output:0-adjust_output1/strided_slice_1/stack:output:0/adjust_output1/strided_slice_1/stack_1:output:0/adjust_output1/strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
ellipsis_mask*
new_axis_mask2 
adjust_output1/strided_slice_1?
$adjust_output1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        2&
$adjust_output1/strided_slice_2/stack?
&adjust_output1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2(
&adjust_output1/strided_slice_2/stack_1?
&adjust_output1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&adjust_output1/strided_slice_2/stack_2?
adjust_output1/strided_slice_2StridedSliceexp/Exp:y:0-adjust_output1/strided_slice_2/stack:output:0/adjust_output1/strided_slice_2/stack_1:output:0/adjust_output1/strided_slice_2/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*
ellipsis_mask*
shrink_axis_mask2 
adjust_output1/strided_slice_2?
adjust_output1/ones_like/ShapeShape'adjust_output1/strided_slice_2:output:0*
T0*
_output_shapes
:2 
adjust_output1/ones_like/Shape?
adjust_output1/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2 
adjust_output1/ones_like/Const?
adjust_output1/ones_likeFill'adjust_output1/ones_like/Shape:output:0'adjust_output1/ones_like/Const:output:0*
T0*#
_output_shapes
:?????????2
adjust_output1/ones_like?
$adjust_output1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"        2&
$adjust_output1/strided_slice_3/stack?
&adjust_output1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2(
&adjust_output1/strided_slice_3/stack_1?
&adjust_output1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&adjust_output1/strided_slice_3/stack_2?
adjust_output1/strided_slice_3StridedSlice!adjust_output1/ones_like:output:0-adjust_output1/strided_slice_3/stack:output:0/adjust_output1/strided_slice_3/stack_1:output:0/adjust_output1/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
ellipsis_mask*
new_axis_mask2 
adjust_output1/strided_slice_3?
$adjust_output1/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB"       2&
$adjust_output1/strided_slice_4/stack?
&adjust_output1/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       2(
&adjust_output1/strided_slice_4/stack_1?
&adjust_output1/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&adjust_output1/strided_slice_4/stack_2?
adjust_output1/strided_slice_4StridedSliceexp/Exp:y:0-adjust_output1/strided_slice_4/stack:output:0/adjust_output1/strided_slice_4/stack_1:output:0/adjust_output1/strided_slice_4/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*
ellipsis_mask*
shrink_axis_mask2 
adjust_output1/strided_slice_4?
$adjust_output1/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB"        2&
$adjust_output1/strided_slice_5/stack?
&adjust_output1/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2(
&adjust_output1/strided_slice_5/stack_1?
&adjust_output1/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&adjust_output1/strided_slice_5/stack_2?
adjust_output1/strided_slice_5StridedSlice'adjust_output1/strided_slice_4:output:0-adjust_output1/strided_slice_5/stack:output:0/adjust_output1/strided_slice_5/stack_1:output:0/adjust_output1/strided_slice_5/stack_2:output:0*
Index0*
T0*'
_output_shapes
:?????????*
ellipsis_mask*
new_axis_mask2 
adjust_output1/strided_slice_5z
adjust_output1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
adjust_output1/concat/axis?
adjust_output1/concatConcatV2'adjust_output1/strided_slice_1:output:0'adjust_output1/strided_slice_3:output:0'adjust_output1/strided_slice_5:output:0#adjust_output1/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????2
adjust_output1/concat?
IdentityIdentityadjust_output1/concat:output:0^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp:N J
+
_output_shapes
:?????????

_user_specified_namex
?	
?
E__inference_dense_2_layer_call_and_return_conditional_losses_25995926

inputs0
matmul_readvariableop_resource:	-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:	*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????	
 
_user_specified_nameinputs
?
F
*__inference_flatten_layer_call_fn_25995861

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_flatten_layer_call_and_return_conditional_losses_259953622
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????	2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
a
E__inference_flatten_layer_call_and_return_conditional_losses_25995867

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????	   2
Constg
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:?????????	2	
Reshaped
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:?????????	2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?	
?
E__inference_dense_2_layer_call_and_return_conditional_losses_25995408

inputs0
matmul_readvariableop_resource:	-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:	*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????	
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
?
input_14
serving_default_input_1:0?????????<
output_10
StatefulPartitionedCall:0?????????tensorflow/serving/predict:ͧ
?
layers_
	optimizer
regularization_losses
	variables
trainable_variables
	keras_api

signatures
k__call__
l_default_save_signature
*m&call_and_return_all_conditional_losses"?	
_tf_keras_model?	{"name": "wb_estimator", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "WbEstimator", "config": {"layer was saved without config": true}, "is_graph_network": false, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 3, 3]}, "float32", "input_1"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "WbEstimator"}, "training_config": {"loss": {"class_name": "MeanAbsoluteError", "config": {"reduction": "auto", "name": "mean_absolute_error"}, "shared_object_id": 0}, "metrics": [[{"class_name": "MeanMetricWrapper", "config": {"name": "ang", "dtype": "float32", "fn": {"class_name": "AngularError", "config": {"reduction": "auto", "name": "ang"}, "__passive_serialization__": true, "shared_object_id": 1}}, "shared_object_id": 2}]], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 6.25000029685907e-05, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
J
0
	1

2
3
4
5"
trackable_list_wrapper
?

beta_1

beta_2
	decay
learning_rate
iterm_m`mambmcmdvevfvgvhvivj"
	optimizer
 "
trackable_list_wrapper
J
0
1
2
3
4
5"
trackable_list_wrapper
J
0
1
2
3
4
5"
trackable_list_wrapper
?
layer_regularization_losses
non_trainable_variables
regularization_losses
	variables
trainable_variables

layers
metrics
layer_metrics
k__call__
l_default_save_signature
*m&call_and_return_all_conditional_losses
&m"call_and_return_conditional_losses"
_generic_user_object
,
nserving_default"
signature_map
?
regularization_losses
	variables
 trainable_variables
!	keras_api
o__call__
*p&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "flatten", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "shared_object_id": 3, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}, "shared_object_id": 4}}
?

kernel
bias
"regularization_losses
#	variables
$trainable_variables
%	keras_api
q__call__
*r&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 9, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 5}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 6}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 7, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 9}}, "shared_object_id": 8}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 9]}}
?

kernel
bias
&regularization_losses
'	variables
(trainable_variables
)	keras_api
s__call__
*t&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "dense_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 9, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 9}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 10}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 11, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 9}}, "shared_object_id": 12}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 9]}}
?

kernel
bias
*regularization_losses
+	variables
,trainable_variables
-	keras_api
u__call__
*v&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "dense_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 2, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 13}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 14}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 15, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 9}}, "shared_object_id": 16}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 9]}}
?
.regularization_losses
/	variables
0trainable_variables
1	keras_api
w__call__
*x&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "exp", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Exp", "config": {"name": "exp", "trainable": true, "dtype": "float32"}, "shared_object_id": 17}
?
2regularization_losses
3	variables
4trainable_variables
5	keras_api
y__call__
*z&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "adjust_output1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "AdjustOutput1", "config": {"name": "adjust_output1", "trainable": true, "dtype": "float32"}, "shared_object_id": 18}
: (2beta_1
: (2beta_2
: (2decay
: (2learning_rate
:	 (2	Adam/iter
+:)		2wb_estimator/dense/kernel
%:#	2wb_estimator/dense/bias
-:+		2wb_estimator/dense_1/kernel
':%	2wb_estimator/dense_1/bias
-:+	2wb_estimator/dense_2/kernel
':%2wb_estimator/dense_2/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
J
0
	1

2
3
4
5"
trackable_list_wrapper
.
60
71"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
8layer_regularization_losses
9non_trainable_variables
regularization_losses
	variables
 trainable_variables

:layers
;metrics
<layer_metrics
o__call__
*p&call_and_return_all_conditional_losses
&p"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
=layer_regularization_losses
>non_trainable_variables
"regularization_losses
#	variables
$trainable_variables

?layers
@metrics
Alayer_metrics
q__call__
*r&call_and_return_all_conditional_losses
&r"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
Blayer_regularization_losses
Cnon_trainable_variables
&regularization_losses
'	variables
(trainable_variables

Dlayers
Emetrics
Flayer_metrics
s__call__
*t&call_and_return_all_conditional_losses
&t"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
Glayer_regularization_losses
Hnon_trainable_variables
*regularization_losses
+	variables
,trainable_variables

Ilayers
Jmetrics
Klayer_metrics
u__call__
*v&call_and_return_all_conditional_losses
&v"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
Llayer_regularization_losses
Mnon_trainable_variables
.regularization_losses
/	variables
0trainable_variables

Nlayers
Ometrics
Player_metrics
w__call__
*x&call_and_return_all_conditional_losses
&x"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
Qlayer_regularization_losses
Rnon_trainable_variables
2regularization_losses
3	variables
4trainable_variables

Slayers
Tmetrics
Ulayer_metrics
y__call__
*z&call_and_return_all_conditional_losses
&z"call_and_return_conditional_losses"
_generic_user_object
?
	Vtotal
	Wcount
X	variables
Y	keras_api"?
_tf_keras_metric?{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}, "shared_object_id": 19}
?
	Ztotal
	[count
\
_fn_kwargs
]	variables
^	keras_api"?
_tf_keras_metric?{"class_name": "MeanMetricWrapper", "name": "ang", "dtype": "float32", "config": {"name": "ang", "dtype": "float32", "fn": {"class_name": "AngularError", "config": {"reduction": "auto", "name": "ang"}, "__passive_serialization__": true, "shared_object_id": 1}}, "shared_object_id": 2}
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
:  (2total
:  (2count
.
V0
W1"
trackable_list_wrapper
-
X	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
Z0
[1"
trackable_list_wrapper
-
]	variables"
_generic_user_object
0:.		2 Adam/wb_estimator/dense/kernel/m
*:(	2Adam/wb_estimator/dense/bias/m
2:0		2"Adam/wb_estimator/dense_1/kernel/m
,:*	2 Adam/wb_estimator/dense_1/bias/m
2:0	2"Adam/wb_estimator/dense_2/kernel/m
,:*2 Adam/wb_estimator/dense_2/bias/m
0:.		2 Adam/wb_estimator/dense/kernel/v
*:(	2Adam/wb_estimator/dense/bias/v
2:0		2"Adam/wb_estimator/dense_1/kernel/v
,:*	2 Adam/wb_estimator/dense_1/bias/v
2:0	2"Adam/wb_estimator/dense_2/kernel/v
,:*2 Adam/wb_estimator/dense_2/bias/v
?2?
/__inference_wb_estimator_layer_call_fn_25995472
/__inference_wb_estimator_layer_call_fn_25995727
/__inference_wb_estimator_layer_call_fn_25995744
/__inference_wb_estimator_layer_call_fn_25995641?
???
FullArgSpec
args?
jself
jx
varargs
 
varkwjkwargs
defaults? 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
?2?
#__inference__wrapped_model_25995349?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? **?'
%?"
input_1?????????
?2?
J__inference_wb_estimator_layer_call_and_return_conditional_losses_25995800
J__inference_wb_estimator_layer_call_and_return_conditional_losses_25995856
J__inference_wb_estimator_layer_call_and_return_conditional_losses_25995663
J__inference_wb_estimator_layer_call_and_return_conditional_losses_25995685?
???
FullArgSpec
args?
jself
jx
varargs
 
varkwjkwargs
defaults? 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
?B?
&__inference_signature_wrapper_25995710input_1"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_flatten_layer_call_fn_25995861?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_flatten_layer_call_and_return_conditional_losses_25995867?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_dense_layer_call_fn_25995876?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_dense_layer_call_and_return_conditional_losses_25995887?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_dense_1_layer_call_fn_25995896?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_dense_1_layer_call_and_return_conditional_losses_25995907?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_dense_2_layer_call_fn_25995916?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_dense_2_layer_call_and_return_conditional_losses_25995926?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
&__inference_exp_layer_call_fn_25995931
&__inference_exp_layer_call_fn_25995936?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkwjkwargs
defaults? 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
?2?
A__inference_exp_layer_call_and_return_conditional_losses_25995941
A__inference_exp_layer_call_and_return_conditional_losses_25995946?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkwjkwargs
defaults? 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
?2?
1__inference_adjust_output1_layer_call_fn_25995951
1__inference_adjust_output1_layer_call_fn_25995956?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkwjkwargs
defaults? 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
?2?
L__inference_adjust_output1_layer_call_and_return_conditional_losses_25995989
L__inference_adjust_output1_layer_call_and_return_conditional_losses_25996022?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkwjkwargs
defaults? 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 ?
#__inference__wrapped_model_25995349s4?1
*?'
%?"
input_1?????????
? "3?0
.
output_1"?
output_1??????????
L__inference_adjust_output1_layer_call_and_return_conditional_losses_25995989h??<
%?"
 ?
inputs?????????
?

trainingp "%?"
?
0?????????
? ?
L__inference_adjust_output1_layer_call_and_return_conditional_losses_25996022h??<
%?"
 ?
inputs?????????
?

trainingp"%?"
?
0?????????
? ?
1__inference_adjust_output1_layer_call_fn_25995951[??<
%?"
 ?
inputs?????????
?

trainingp "???????????
1__inference_adjust_output1_layer_call_fn_25995956[??<
%?"
 ?
inputs?????????
?

trainingp"???????????
E__inference_dense_1_layer_call_and_return_conditional_losses_25995907\/?,
%?"
 ?
inputs?????????	
? "%?"
?
0?????????	
? }
*__inference_dense_1_layer_call_fn_25995896O/?,
%?"
 ?
inputs?????????	
? "??????????	?
E__inference_dense_2_layer_call_and_return_conditional_losses_25995926\/?,
%?"
 ?
inputs?????????	
? "%?"
?
0?????????
? }
*__inference_dense_2_layer_call_fn_25995916O/?,
%?"
 ?
inputs?????????	
? "???????????
C__inference_dense_layer_call_and_return_conditional_losses_25995887\/?,
%?"
 ?
inputs?????????	
? "%?"
?
0?????????	
? {
(__inference_dense_layer_call_fn_25995876O/?,
%?"
 ?
inputs?????????	
? "??????????	?
A__inference_exp_layer_call_and_return_conditional_losses_25995941h??<
%?"
 ?
inputs?????????
?

trainingp "%?"
?
0?????????
? ?
A__inference_exp_layer_call_and_return_conditional_losses_25995946h??<
%?"
 ?
inputs?????????
?

trainingp"%?"
?
0?????????
? ?
&__inference_exp_layer_call_fn_25995931[??<
%?"
 ?
inputs?????????
?

trainingp "???????????
&__inference_exp_layer_call_fn_25995936[??<
%?"
 ?
inputs?????????
?

trainingp"???????????
E__inference_flatten_layer_call_and_return_conditional_losses_25995867\3?0
)?&
$?!
inputs?????????
? "%?"
?
0?????????	
? }
*__inference_flatten_layer_call_fn_25995861O3?0
)?&
$?!
inputs?????????
? "??????????	?
&__inference_signature_wrapper_25995710~??<
? 
5?2
0
input_1%?"
input_1?????????"3?0
.
output_1"?
output_1??????????
J__inference_wb_estimator_layer_call_and_return_conditional_losses_25995663uD?A
*?'
%?"
input_1?????????
?

trainingp "%?"
?
0?????????
? ?
J__inference_wb_estimator_layer_call_and_return_conditional_losses_25995685uD?A
*?'
%?"
input_1?????????
?

trainingp"%?"
?
0?????????
? ?
J__inference_wb_estimator_layer_call_and_return_conditional_losses_25995800o>?;
$?!
?
x?????????
?

trainingp "%?"
?
0?????????
? ?
J__inference_wb_estimator_layer_call_and_return_conditional_losses_25995856o>?;
$?!
?
x?????????
?

trainingp"%?"
?
0?????????
? ?
/__inference_wb_estimator_layer_call_fn_25995472hD?A
*?'
%?"
input_1?????????
?

trainingp "???????????
/__inference_wb_estimator_layer_call_fn_25995641hD?A
*?'
%?"
input_1?????????
?

trainingp"???????????
/__inference_wb_estimator_layer_call_fn_25995727b>?;
$?!
?
x?????????
?

trainingp "???????????
/__inference_wb_estimator_layer_call_fn_25995744b>?;
$?!
?
x?????????
?

trainingp"??????????