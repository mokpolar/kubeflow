ва
§
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
О
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
executor_typestring 
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"serve*2.0.02unknown8ЅЙ
v
dense/kernelVarHandleOp*
_output_shapes
: *
shape:
*
shared_namedense/kernel*
dtype0
o
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel* 
_output_shapes
:
*
dtype0
m

dense/biasVarHandleOp*
shared_name
dense/bias*
dtype0*
_output_shapes
: *
shape:
f
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
dtype0*
_output_shapes	
:
y
dense_1/kernelVarHandleOp*
shape:	
*
shared_namedense_1/kernel*
dtype0*
_output_shapes
: 
r
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel*
dtype0*
_output_shapes
:	

p
dense_1/biasVarHandleOp*
_output_shapes
: *
shape:
*
shared_namedense_1/bias*
dtype0
i
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
dtype0*
_output_shapes
:

d
SGD/iterVarHandleOp*
shared_name
SGD/iter*
dtype0	*
_output_shapes
: *
shape: 
]
SGD/iter/Read/ReadVariableOpReadVariableOpSGD/iter*
dtype0	*
_output_shapes
: 
f
	SGD/decayVarHandleOp*
shape: *
shared_name	SGD/decay*
dtype0*
_output_shapes
: 
_
SGD/decay/Read/ReadVariableOpReadVariableOp	SGD/decay*
dtype0*
_output_shapes
: 
v
SGD/learning_rateVarHandleOp*"
shared_nameSGD/learning_rate*
dtype0*
_output_shapes
: *
shape: 
o
%SGD/learning_rate/Read/ReadVariableOpReadVariableOpSGD/learning_rate*
_output_shapes
: *
dtype0
l
SGD/momentumVarHandleOp*
shared_nameSGD/momentum*
dtype0*
_output_shapes
: *
shape: 
e
 SGD/momentum/Read/ReadVariableOpReadVariableOpSGD/momentum*
dtype0*
_output_shapes
: 
^
totalVarHandleOp*
shared_nametotal*
dtype0*
_output_shapes
: *
shape: 
W
total/Read/ReadVariableOpReadVariableOptotal*
dtype0*
_output_shapes
: 
^
countVarHandleOp*
shape: *
shared_namecount*
dtype0*
_output_shapes
: 
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0

NoOpNoOp
ъ
ConstConst"/device:CPU:0*Ѕ
valueB B
ц
layer-0
layer-1
layer_with_weights-0
layer-2
layer-3
layer_with_weights-1
layer-4
	optimizer
regularization_losses
trainable_variables
		variables

	keras_api

signatures
R
regularization_losses
trainable_variables
	variables
	keras_api
R
regularization_losses
trainable_variables
	variables
	keras_api
h

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
R
regularization_losses
trainable_variables
	variables
	keras_api
h

kernel
bias
 regularization_losses
!trainable_variables
"	variables
#	keras_api
6
$iter
	%decay
&learning_rate
'momentum
 

0
1
2
3

0
1
2
3

(layer_regularization_losses
regularization_losses
)metrics

*layers
trainable_variables
+non_trainable_variables
		variables
 
 
 
 

,layer_regularization_losses
regularization_losses
-metrics

.layers
trainable_variables
/non_trainable_variables
	variables
 
 
 

0layer_regularization_losses
regularization_losses
1metrics

2layers
trainable_variables
3non_trainable_variables
	variables
XV
VARIABLE_VALUEdense/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
dense/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1

4layer_regularization_losses
regularization_losses
5metrics

6layers
trainable_variables
7non_trainable_variables
	variables
 
 
 

8layer_regularization_losses
regularization_losses
9metrics

:layers
trainable_variables
;non_trainable_variables
	variables
ZX
VARIABLE_VALUEdense_1/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_1/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1

<layer_regularization_losses
 regularization_losses
=metrics

>layers
!trainable_variables
?non_trainable_variables
"	variables
GE
VARIABLE_VALUESGD/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
IG
VARIABLE_VALUE	SGD/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUESGD/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUESGD/momentum-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
 

@0

0
1
2
3
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
x
	Atotal
	Bcount
C
_fn_kwargs
Dregularization_losses
Etrainable_variables
F	variables
G	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE
 
 
 

A0
B1

Hlayer_regularization_losses
Dregularization_losses
Imetrics

Jlayers
Etrainable_variables
Knon_trainable_variables
F	variables
 
 
 

A0
B1*
dtype0*
_output_shapes
: 

serving_default_flatten_inputPlaceholder*
dtype0*+
_output_shapes
:џџџџџџџџџ* 
shape:џџџџџџџџџ
л
StatefulPartitionedCallStatefulPartitionedCallserving_default_flatten_inputdense/kernel
dense/biasdense_1/kerneldense_1/bias*,
f'R%
#__inference_signature_wrapper_29439*
Tout
2**
config_proto

GPU 

CPU2J 8*
Tin	
2*'
_output_shapes
:џџџџџџџџџ
*,
_gradient_op_typePartitionedCall-29622
O
saver_filenamePlaceholder*
dtype0*
_output_shapes
: *
shape: 
в
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOp"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOpSGD/iter/Read/ReadVariableOpSGD/decay/Read/ReadVariableOp%SGD/learning_rate/Read/ReadVariableOp SGD/momentum/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOpConst*'
f"R 
__inference__traced_save_29653*
Tout
2**
config_proto

GPU 

CPU2J 8*
_output_shapes
: *
Tin
2	*,
_gradient_op_typePartitionedCall-29654

StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense/kernel
dense/biasdense_1/kerneldense_1/biasSGD/iter	SGD/decaySGD/learning_rateSGD/momentumtotalcount*,
_gradient_op_typePartitionedCall-29697**
f%R#
!__inference__traced_restore_29696*
Tout
2**
config_proto

GPU 

CPU2J 8*
_output_shapes
: *
Tin
2П
ћ
ѓ
*__inference_sequential_layer_call_fn_29516

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identityЂStatefulPartitionedCallЎ
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*
Tin	
2*'
_output_shapes
:џџџџџџџџџ
*,
_gradient_op_typePartitionedCall-29418*N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_29417*
Tout
2**
config_proto

GPU 

CPU2J 8
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ
"
identityIdentity:output:0*:
_input_shapes)
':џџџџџџџџџ::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : : : 
З
Ъ
 __inference__wrapped_model_29238
flatten_input3
/sequential_dense_matmul_readvariableop_resource4
0sequential_dense_biasadd_readvariableop_resource5
1sequential_dense_1_matmul_readvariableop_resource6
2sequential_dense_1_biasadd_readvariableop_resource
identityЂ'sequential/dense/BiasAdd/ReadVariableOpЂ&sequential/dense/MatMul/ReadVariableOpЂ)sequential/dense_1/BiasAdd/ReadVariableOpЂ(sequential/dense_1/MatMul/ReadVariableOpq
 sequential/flatten/Reshape/shapeConst*
valueB"џџџџ  *
dtype0*
_output_shapes
:
sequential/flatten/ReshapeReshapeflatten_input)sequential/flatten/Reshape/shape:output:0*(
_output_shapes
:џџџџџџџџџ*
T0Ц
&sequential/dense/MatMul/ReadVariableOpReadVariableOp/sequential_dense_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0* 
_output_shapes
:
Љ
sequential/dense/MatMulMatMul#sequential/flatten/Reshape:output:0.sequential/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџУ
'sequential/dense/BiasAdd/ReadVariableOpReadVariableOp0sequential_dense_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes	
:*
dtype0Њ
sequential/dense/BiasAddBiasAdd!sequential/dense/MatMul:product:0/sequential/dense/BiasAdd/ReadVariableOp:value:0*(
_output_shapes
:џџџџџџџџџ*
T0s
sequential/dense/ReluRelu!sequential/dense/BiasAdd:output:0*(
_output_shapes
:џџџџџџџџџ*
T0
sequential/dropout/IdentityIdentity#sequential/dense/Relu:activations:0*
T0*(
_output_shapes
:џџџџџџџџџЩ
(sequential/dense_1/MatMul/ReadVariableOpReadVariableOp1sequential_dense_1_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	
­
sequential/dense_1/MatMulMatMul$sequential/dropout/Identity:output:00sequential/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
Ц
)sequential/dense_1/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_1_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:
*
dtype0Џ
sequential/dense_1/BiasAddBiasAdd#sequential/dense_1/MatMul:product:01sequential/dense_1/BiasAdd/ReadVariableOp:value:0*'
_output_shapes
:џџџџџџџџџ
*
T0|
sequential/dense_1/SoftmaxSoftmax#sequential/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ

IdentityIdentity$sequential/dense_1/Softmax:softmax:0(^sequential/dense/BiasAdd/ReadVariableOp'^sequential/dense/MatMul/ReadVariableOp*^sequential/dense_1/BiasAdd/ReadVariableOp)^sequential/dense_1/MatMul/ReadVariableOp*'
_output_shapes
:џџџџџџџџџ
*
T0"
identityIdentity:output:0*:
_input_shapes)
':џџџџџџџџџ::::2P
&sequential/dense/MatMul/ReadVariableOp&sequential/dense/MatMul/ReadVariableOp2R
'sequential/dense/BiasAdd/ReadVariableOp'sequential/dense/BiasAdd/ReadVariableOp2V
)sequential/dense_1/BiasAdd/ReadVariableOp)sequential/dense_1/BiasAdd/ReadVariableOp2T
(sequential/dense_1/MatMul/ReadVariableOp(sequential/dense_1/MatMul/ReadVariableOp: : : :- )
'
_user_specified_nameflatten_input: 

`
B__inference_dropout_layer_call_and_return_conditional_losses_29318

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:џџџџџџџџџ\

Identity_1IdentityIdentity:output:0*(
_output_shapes
:џџџџџџџџџ*
T0"!

identity_1Identity_1:output:0*'
_input_shapes
:џџџџџџџџџ:& "
 
_user_specified_nameinputs
х%

E__inference_sequential_layer_call_and_return_conditional_losses_29477

inputs(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource
identityЂdense/BiasAdd/ReadVariableOpЂdense/MatMul/ReadVariableOpЂdense_1/BiasAdd/ReadVariableOpЂdense_1/MatMul/ReadVariableOpf
flatten/Reshape/shapeConst*
_output_shapes
:*
valueB"џџџџ  *
dtype0u
flatten/ReshapeReshapeinputsflatten/Reshape/shape:output:0*(
_output_shapes
:џџџџџџџџџ*
T0А
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0* 
_output_shapes
:

dense/MatMulMatMulflatten/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ­
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*(
_output_shapes
:џџџџџџџџџ*
T0]

dense/ReluReludense/BiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџY
dropout/dropout/rateConst*
valueB
 *ЭЬL>*
dtype0*
_output_shapes
: ]
dropout/dropout/ShapeShapedense/Relu:activations:0*
T0*
_output_shapes
:g
"dropout/dropout/random_uniform/minConst*
valueB
 *    *
dtype0*
_output_shapes
: g
"dropout/dropout/random_uniform/maxConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
dtype0*(
_output_shapes
:џџџџџџџџџ*
T0Є
"dropout/dropout/random_uniform/subSub+dropout/dropout/random_uniform/max:output:0+dropout/dropout/random_uniform/min:output:0*
_output_shapes
: *
T0Л
"dropout/dropout/random_uniform/mulMul5dropout/dropout/random_uniform/RandomUniform:output:0&dropout/dropout/random_uniform/sub:z:0*(
_output_shapes
:џџџџџџџџџ*
T0­
dropout/dropout/random_uniformAdd&dropout/dropout/random_uniform/mul:z:0+dropout/dropout/random_uniform/min:output:0*
T0*(
_output_shapes
:џџџџџџџџџZ
dropout/dropout/sub/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: z
dropout/dropout/subSubdropout/dropout/sub/x:output:0dropout/dropout/rate:output:0*
T0*
_output_shapes
: ^
dropout/dropout/truediv/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
dropout/dropout/truedivRealDiv"dropout/dropout/truediv/x:output:0dropout/dropout/sub:z:0*
_output_shapes
: *
T0Ђ
dropout/dropout/GreaterEqualGreaterEqual"dropout/dropout/random_uniform:z:0dropout/dropout/rate:output:0*(
_output_shapes
:џџџџџџџџџ*
T0
dropout/dropout/mulMuldense/Relu:activations:0dropout/dropout/truediv:z:0*
T0*(
_output_shapes
:џџџџџџџџџ
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*(
_output_shapes
:џџџџџџџџџ*

SrcT0

dropout/dropout/mul_1Muldropout/dropout/mul:z:0dropout/dropout/Cast:y:0*
T0*(
_output_shapes
:џџџџџџџџџГ
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	

dense_1/MatMulMatMuldropout/dropout/mul_1:z:0%dense_1/MatMul/ReadVariableOp:value:0*'
_output_shapes
:џџџџџџџџџ
*
T0А
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:

dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
f
dense_1/SoftmaxSoftmaxdense_1/BiasAdd:output:0*'
_output_shapes
:џџџџџџџџџ
*
T0п
IdentityIdentitydense_1/Softmax:softmax:0^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ
"
identityIdentity:output:0*:
_input_shapes)
':џџџџџџџџџ::::2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp: :& "
 
_user_specified_nameinputs: : : 
ѕ
^
B__inference_flatten_layer_call_and_return_conditional_losses_29522

inputs
identity^
Reshape/shapeConst*
valueB"џџџџ  *
dtype0*
_output_shapes
:e
ReshapeReshapeinputsReshape/shape:output:0*(
_output_shapes
:џџџџџџџџџ*
T0Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0**
_input_shapes
:џџџџџџџџџ:& "
 
_user_specified_nameinputs
Ф
в
E__inference_sequential_layer_call_and_return_conditional_losses_29417

inputs(
$dense_statefulpartitionedcall_args_1(
$dense_statefulpartitionedcall_args_2*
&dense_1_statefulpartitionedcall_args_1*
&dense_1_statefulpartitionedcall_args_2
identityЂdense/StatefulPartitionedCallЂdense_1/StatefulPartitionedCall 
flatten/PartitionedCallPartitionedCallinputs*,
_gradient_op_typePartitionedCall-29256*K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_29250*
Tout
2**
config_proto

GPU 

CPU2J 8*(
_output_shapes
:џџџџџџџџџ*
Tin
2
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0$dense_statefulpartitionedcall_args_1$dense_statefulpartitionedcall_args_2*,
_gradient_op_typePartitionedCall-29280*I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_29274*
Tout
2**
config_proto

GPU 

CPU2J 8*
Tin
2*(
_output_shapes
:џџџџџџџџџР
dropout/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*,
_gradient_op_typePartitionedCall-29330*K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_29318*
Tout
2**
config_proto

GPU 

CPU2J 8*(
_output_shapes
:џџџџџџџџџ*
Tin
2
dense_1/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0&dense_1_statefulpartitionedcall_args_1&dense_1_statefulpartitionedcall_args_2*K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_29346*
Tout
2**
config_proto

GPU 

CPU2J 8*
Tin
2*'
_output_shapes
:џџџџџџџџџ
*,
_gradient_op_typePartitionedCall-29352В
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*'
_output_shapes
:џџџџџџџџџ
*
T0"
identityIdentity:output:0*:
_input_shapes)
':џџџџџџџџџ::::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall:& "
 
_user_specified_nameinputs: : : : 
ћ
ѓ
*__inference_sequential_layer_call_fn_29507

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identityЂStatefulPartitionedCallЎ
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*,
_gradient_op_typePartitionedCall-29394*N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_29393*
Tout
2**
config_proto

GPU 

CPU2J 8*'
_output_shapes
:џџџџџџџџџ
*
Tin	
2
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*'
_output_shapes
:џџџџџџџџџ
*
T0"
identityIdentity:output:0*:
_input_shapes)
':џџџџџџџџџ::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : : : 
в
І
%__inference_dense_layer_call_fn_29545

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identityЂStatefulPartitionedCallш
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2**
config_proto

GPU 

CPU2J 8*(
_output_shapes
:џџџџџџџџџ*
Tin
2*,
_gradient_op_typePartitionedCall-29280*I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_29274*
Tout
2
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*(
_output_shapes
:џџџџџџџџџ*
T0"
identityIdentity:output:0*/
_input_shapes
:џџџџџџџџџ::22
StatefulPartitionedCallStatefulPartitionedCall: : :& "
 
_user_specified_nameinputs
Џ
a
B__inference_dropout_layer_call_and_return_conditional_losses_29565

inputs
identityQ
dropout/rateConst*
valueB
 *ЭЬL>*
dtype0*
_output_shapes
: C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:_
dropout/random_uniform/minConst*
valueB
 *    *
dtype0*
_output_shapes
: _
dropout/random_uniform/maxConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*
dtype0*(
_output_shapes
:џџџџџџџџџ
dropout/random_uniform/subSub#dropout/random_uniform/max:output:0#dropout/random_uniform/min:output:0*
T0*
_output_shapes
: Ѓ
dropout/random_uniform/mulMul-dropout/random_uniform/RandomUniform:output:0dropout/random_uniform/sub:z:0*
T0*(
_output_shapes
:џџџџџџџџџ
dropout/random_uniformAdddropout/random_uniform/mul:z:0#dropout/random_uniform/min:output:0*
T0*(
_output_shapes
:џџџџџџџџџR
dropout/sub/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: b
dropout/subSubdropout/sub/x:output:0dropout/rate:output:0*
_output_shapes
: *
T0V
dropout/truediv/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: h
dropout/truedivRealDivdropout/truediv/x:output:0dropout/sub:z:0*
_output_shapes
: *
T0
dropout/GreaterEqualGreaterEqualdropout/random_uniform:z:0dropout/rate:output:0*(
_output_shapes
:џџџџџџџџџ*
T0b
dropout/mulMulinputsdropout/truediv:z:0*
T0*(
_output_shapes
:џџџџџџџџџp
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*(
_output_shapes
:џџџџџџџџџ*

SrcT0
j
dropout/mul_1Muldropout/mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:џџџџџџџџџZ
IdentityIdentitydropout/mul_1:z:0*
T0*(
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*'
_input_shapes
:џџџџџџџџџ:& "
 
_user_specified_nameinputs
д	
й
@__inference_dense_layer_call_and_return_conditional_losses_29274

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpЄ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0* 
_output_shapes
:
j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЁ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*(
_output_shapes
:џџџџџџџџџ*
T0Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*/
_input_shapes
:џџџџџџџџџ::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp: :& "
 
_user_specified_nameinputs: 
љ
ћ
E__inference_sequential_layer_call_and_return_conditional_losses_29364
flatten_input(
$dense_statefulpartitionedcall_args_1(
$dense_statefulpartitionedcall_args_2*
&dense_1_statefulpartitionedcall_args_1*
&dense_1_statefulpartitionedcall_args_2
identityЂdense/StatefulPartitionedCallЂdense_1/StatefulPartitionedCallЂdropout/StatefulPartitionedCallЇ
flatten/PartitionedCallPartitionedCallflatten_input**
config_proto

GPU 

CPU2J 8*(
_output_shapes
:џџџџџџџџџ*
Tin
2*,
_gradient_op_typePartitionedCall-29256*K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_29250*
Tout
2
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0$dense_statefulpartitionedcall_args_1$dense_statefulpartitionedcall_args_2*,
_gradient_op_typePartitionedCall-29280*I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_29274*
Tout
2**
config_proto

GPU 

CPU2J 8*
Tin
2*(
_output_shapes
:џџџџџџџџџа
dropout/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0*,
_gradient_op_typePartitionedCall-29322*K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_29311*
Tout
2**
config_proto

GPU 

CPU2J 8*
Tin
2*(
_output_shapes
:џџџџџџџџџЃ
dense_1/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0&dense_1_statefulpartitionedcall_args_1&dense_1_statefulpartitionedcall_args_2*,
_gradient_op_typePartitionedCall-29352*K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_29346*
Tout
2**
config_proto

GPU 

CPU2J 8*
Tin
2*'
_output_shapes
:џџџџџџџџџ
д
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dropout/StatefulPartitionedCall*'
_output_shapes
:џџџџџџџџџ
*
T0"
identityIdentity:output:0*:
_input_shapes)
':џџџџџџџџџ::::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall: : : : :- )
'
_user_specified_nameflatten_input

њ
*__inference_sequential_layer_call_fn_29401
flatten_input"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identityЂStatefulPartitionedCallЕ
StatefulPartitionedCallStatefulPartitionedCallflatten_inputstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*'
_output_shapes
:џџџџџџџџџ
*
Tin	
2*,
_gradient_op_typePartitionedCall-29394*N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_29393*
Tout
2**
config_proto

GPU 

CPU2J 8
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ
"
identityIdentity:output:0*:
_input_shapes)
':џџџџџџџџџ::::22
StatefulPartitionedCallStatefulPartitionedCall: : : : :- )
'
_user_specified_nameflatten_input
 

E__inference_sequential_layer_call_and_return_conditional_losses_29498

inputs(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource
identityЂdense/BiasAdd/ReadVariableOpЂdense/MatMul/ReadVariableOpЂdense_1/BiasAdd/ReadVariableOpЂdense_1/MatMul/ReadVariableOpf
flatten/Reshape/shapeConst*
valueB"џџџџ  *
dtype0*
_output_shapes
:u
flatten/ReshapeReshapeinputsflatten/Reshape/shape:output:0*
T0*(
_output_shapes
:џџџџџџџџџА
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0* 
_output_shapes
:
*
dtype0
dense/MatMulMatMulflatten/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*(
_output_shapes
:џџџџџџџџџ*
T0­
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ]

dense/ReluReludense/BiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџi
dropout/IdentityIdentitydense/Relu:activations:0*
T0*(
_output_shapes
:џџџџџџџџџГ
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	

dense_1/MatMulMatMuldropout/Identity:output:0%dense_1/MatMul/ReadVariableOp:value:0*'
_output_shapes
:џџџџџџџџџ
*
T0А
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:

dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
f
dense_1/SoftmaxSoftmaxdense_1/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ
п
IdentityIdentitydense_1/Softmax:softmax:0^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp*'
_output_shapes
:џџџџџџџџџ
*
T0"
identityIdentity:output:0*:
_input_shapes)
':џџџџџџџџџ::::2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp:& "
 
_user_specified_nameinputs: : : : 
ё*

!__inference__traced_restore_29696
file_prefix!
assignvariableop_dense_kernel!
assignvariableop_1_dense_bias%
!assignvariableop_2_dense_1_kernel#
assignvariableop_3_dense_1_bias
assignvariableop_4_sgd_iter 
assignvariableop_5_sgd_decay(
$assignvariableop_6_sgd_learning_rate#
assignvariableop_7_sgd_momentum
assignvariableop_8_total
assignvariableop_9_count
identity_11ЂAssignVariableOpЂAssignVariableOp_1ЂAssignVariableOp_2ЂAssignVariableOp_3ЂAssignVariableOp_4ЂAssignVariableOp_5ЂAssignVariableOp_6ЂAssignVariableOp_7ЂAssignVariableOp_8ЂAssignVariableOp_9Ђ	RestoreV2ЂRestoreV2_1ё
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:
*
valueB
B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*
dtype0
RestoreV2/shape_and_slicesConst"/device:CPU:0*'
valueB
B B B B B B B B B B *
dtype0*
_output_shapes
:
а
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*<
_output_shapes*
(::::::::::*
dtypes
2
	L
IdentityIdentityRestoreV2:tensors:0*
_output_shapes
:*
T0y
AssignVariableOpAssignVariableOpassignvariableop_dense_kernelIdentity:output:0*
dtype0*
_output_shapes
 N

Identity_1IdentityRestoreV2:tensors:1*
_output_shapes
:*
T0}
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_biasIdentity_1:output:0*
dtype0*
_output_shapes
 N

Identity_2IdentityRestoreV2:tensors:2*
_output_shapes
:*
T0
AssignVariableOp_2AssignVariableOp!assignvariableop_2_dense_1_kernelIdentity_2:output:0*
dtype0*
_output_shapes
 N

Identity_3IdentityRestoreV2:tensors:3*
_output_shapes
:*
T0
AssignVariableOp_3AssignVariableOpassignvariableop_3_dense_1_biasIdentity_3:output:0*
dtype0*
_output_shapes
 N

Identity_4IdentityRestoreV2:tensors:4*
_output_shapes
:*
T0	{
AssignVariableOp_4AssignVariableOpassignvariableop_4_sgd_iterIdentity_4:output:0*
dtype0	*
_output_shapes
 N

Identity_5IdentityRestoreV2:tensors:5*
T0*
_output_shapes
:|
AssignVariableOp_5AssignVariableOpassignvariableop_5_sgd_decayIdentity_5:output:0*
dtype0*
_output_shapes
 N

Identity_6IdentityRestoreV2:tensors:6*
T0*
_output_shapes
:
AssignVariableOp_6AssignVariableOp$assignvariableop_6_sgd_learning_rateIdentity_6:output:0*
dtype0*
_output_shapes
 N

Identity_7IdentityRestoreV2:tensors:7*
T0*
_output_shapes
:
AssignVariableOp_7AssignVariableOpassignvariableop_7_sgd_momentumIdentity_7:output:0*
dtype0*
_output_shapes
 N

Identity_8IdentityRestoreV2:tensors:8*
T0*
_output_shapes
:x
AssignVariableOp_8AssignVariableOpassignvariableop_8_totalIdentity_8:output:0*
dtype0*
_output_shapes
 N

Identity_9IdentityRestoreV2:tensors:9*
T0*
_output_shapes
:x
AssignVariableOp_9AssignVariableOpassignvariableop_9_countIdentity_9:output:0*
_output_shapes
 *
dtype0
RestoreV2_1/tensor_namesConst"/device:CPU:0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH*
dtype0*
_output_shapes
:t
RestoreV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
valueB
B *
dtype0Е
RestoreV2_1	RestoreV2file_prefix!RestoreV2_1/tensor_names:output:0%RestoreV2_1/shape_and_slices:output:0
^RestoreV2"/device:CPU:0*
_output_shapes
:*
dtypes
21
NoOpNoOp"/device:CPU:0*
_output_shapes
 Ћ
Identity_10Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
_output_shapes
: *
T0И
Identity_11IdentityIdentity_10:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9
^RestoreV2^RestoreV2_1*
_output_shapes
: *
T0"#
identity_11Identity_11:output:0*=
_input_shapes,
*: ::::::::::2
RestoreV2_1RestoreV2_12(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92
	RestoreV2	RestoreV2:	 :
 :+ '
%
_user_specified_namefile_prefix: : : : : : : : 

`
B__inference_dropout_layer_call_and_return_conditional_losses_29570

inputs

identity_1O
IdentityIdentityinputs*(
_output_shapes
:џџџџџџџџџ*
T0\

Identity_1IdentityIdentity:output:0*(
_output_shapes
:џџџџџџџџџ*
T0"!

identity_1Identity_1:output:0*'
_input_shapes
:џџџџџџџџџ:& "
 
_user_specified_nameinputs
Ж
C
'__inference_flatten_layer_call_fn_29527

inputs
identity
PartitionedCallPartitionedCallinputs**
config_proto

GPU 

CPU2J 8*
Tin
2*(
_output_shapes
:џџџџџџџџџ*,
_gradient_op_typePartitionedCall-29256*K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_29250*
Tout
2a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0**
_input_shapes
:џџџџџџџџџ:& "
 
_user_specified_nameinputs
Г
C
'__inference_dropout_layer_call_fn_29580

inputs
identity
PartitionedCallPartitionedCallinputs*K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_29318*
Tout
2**
config_proto

GPU 

CPU2J 8*(
_output_shapes
:џџџџџџџџџ*
Tin
2*,
_gradient_op_typePartitionedCall-29330a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*'
_input_shapes
:џџџџџџџџџ:& "
 
_user_specified_nameinputs
е	
л
B__inference_dense_1_layer_call_and_return_conditional_losses_29591

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpЃ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	
i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
 
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:
v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*'
_output_shapes
:џџџџџџџџџ
*
T0V
SoftmaxSoftmaxBiasAdd:output:0*'
_output_shapes
:џџџџџџџџџ
*
T0
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:џџџџџџџџџ
"
identityIdentity:output:0*/
_input_shapes
:џџџџџџџџџ::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp:& "
 
_user_specified_nameinputs: : 
ф
ѓ
#__inference_signature_wrapper_29439
flatten_input"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallflatten_inputstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4**
config_proto

GPU 

CPU2J 8*
Tin	
2*'
_output_shapes
:џџџџџџџџџ
*,
_gradient_op_typePartitionedCall-29432*)
f$R"
 __inference__wrapped_model_29238*
Tout
2
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ
"
identityIdentity:output:0*:
_input_shapes)
':џџџџџџџџџ::::22
StatefulPartitionedCallStatefulPartitionedCall:- )
'
_user_specified_nameflatten_input: : : : 
Џ
a
B__inference_dropout_layer_call_and_return_conditional_losses_29311

inputs
identityQ
dropout/rateConst*
valueB
 *ЭЬL>*
dtype0*
_output_shapes
: C
dropout/ShapeShapeinputs*
T0*
_output_shapes
:_
dropout/random_uniform/minConst*
valueB
 *    *
dtype0*
_output_shapes
: _
dropout/random_uniform/maxConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*
dtype0*(
_output_shapes
:џџџџџџџџџ
dropout/random_uniform/subSub#dropout/random_uniform/max:output:0#dropout/random_uniform/min:output:0*
T0*
_output_shapes
: Ѓ
dropout/random_uniform/mulMul-dropout/random_uniform/RandomUniform:output:0dropout/random_uniform/sub:z:0*
T0*(
_output_shapes
:џџџџџџџџџ
dropout/random_uniformAdddropout/random_uniform/mul:z:0#dropout/random_uniform/min:output:0*
T0*(
_output_shapes
:џџџџџџџџџR
dropout/sub/xConst*
_output_shapes
: *
valueB
 *  ?*
dtype0b
dropout/subSubdropout/sub/x:output:0dropout/rate:output:0*
T0*
_output_shapes
: V
dropout/truediv/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: h
dropout/truedivRealDivdropout/truediv/x:output:0dropout/sub:z:0*
T0*
_output_shapes
: 
dropout/GreaterEqualGreaterEqualdropout/random_uniform:z:0dropout/rate:output:0*(
_output_shapes
:џџџџџџџџџ*
T0b
dropout/mulMulinputsdropout/truediv:z:0*(
_output_shapes
:џџџџџџџџџ*
T0p
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*(
_output_shapes
:џџџџџџџџџ*

SrcT0
j
dropout/mul_1Muldropout/mul:z:0dropout/Cast:y:0*(
_output_shapes
:џџџџџџџџџ*
T0Z
IdentityIdentitydropout/mul_1:z:0*(
_output_shapes
:џџџџџџџџџ*
T0"
identityIdentity:output:0*'
_input_shapes
:џџџџџџџџџ:& "
 
_user_specified_nameinputs
ф
є
E__inference_sequential_layer_call_and_return_conditional_losses_29393

inputs(
$dense_statefulpartitionedcall_args_1(
$dense_statefulpartitionedcall_args_2*
&dense_1_statefulpartitionedcall_args_1*
&dense_1_statefulpartitionedcall_args_2
identityЂdense/StatefulPartitionedCallЂdense_1/StatefulPartitionedCallЂdropout/StatefulPartitionedCall 
flatten/PartitionedCallPartitionedCallinputs*,
_gradient_op_typePartitionedCall-29256*K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_29250*
Tout
2**
config_proto

GPU 

CPU2J 8*(
_output_shapes
:џџџџџџџџџ*
Tin
2
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0$dense_statefulpartitionedcall_args_1$dense_statefulpartitionedcall_args_2*I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_29274*
Tout
2**
config_proto

GPU 

CPU2J 8*
Tin
2*(
_output_shapes
:џџџџџџџџџ*,
_gradient_op_typePartitionedCall-29280а
dropout/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0**
config_proto

GPU 

CPU2J 8*
Tin
2*(
_output_shapes
:џџџџџџџџџ*,
_gradient_op_typePartitionedCall-29322*K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_29311*
Tout
2Ѓ
dense_1/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0&dense_1_statefulpartitionedcall_args_1&dense_1_statefulpartitionedcall_args_2**
config_proto

GPU 

CPU2J 8*'
_output_shapes
:џџџџџџџџџ
*
Tin
2*,
_gradient_op_typePartitionedCall-29352*K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_29346*
Tout
2д
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dropout/StatefulPartitionedCall*'
_output_shapes
:џџџџџџџџџ
*
T0"
identityIdentity:output:0*:
_input_shapes)
':џџџџџџџџџ::::2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall:& "
 
_user_specified_nameinputs: : : : 
н
Г
__inference__traced_save_29653
file_prefix+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop-
)savev2_dense_1_kernel_read_readvariableop+
'savev2_dense_1_bias_read_readvariableop'
#savev2_sgd_iter_read_readvariableop	(
$savev2_sgd_decay_read_readvariableop0
,savev2_sgd_learning_rate_read_readvariableop+
'savev2_sgd_momentum_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop
savev2_1_const

identity_1ЂMergeV2CheckpointsЂSaveV2ЂSaveV2_1
StringJoin/inputs_1Const"/device:CPU:0*<
value3B1 B+_temp_5190cf5838364e3bb8114178c936533a/part*
dtype0*
_output_shapes
: s

StringJoin
StringJoinfile_prefixStringJoin/inputs_1:output:0"/device:CPU:0*
N*
_output_shapes
: L

num_shardsConst*
value	B :*
dtype0*
_output_shapes
: f
ShardedFilename/shardConst"/device:CPU:0*
value	B : *
dtype0*
_output_shapes
: 
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ю
SaveV2/tensor_namesConst"/device:CPU:0*
valueB
B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:

SaveV2/shape_and_slicesConst"/device:CPU:0*'
valueB
B B B B B B B B B B *
dtype0*
_output_shapes
:
Е
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableop#savev2_sgd_iter_read_readvariableop$savev2_sgd_decay_read_readvariableop,savev2_sgd_learning_rate_read_readvariableop'savev2_sgd_momentum_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"/device:CPU:0*
_output_shapes
 *
dtypes
2
	h
ShardedFilename_1/shardConst"/device:CPU:0*
_output_shapes
: *
value	B :*
dtype0
ShardedFilename_1ShardedFilenameStringJoin:output:0 ShardedFilename_1/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 
SaveV2_1/tensor_namesConst"/device:CPU:0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH*
dtype0*
_output_shapes
:q
SaveV2_1/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:У
SaveV2_1SaveV2ShardedFilename_1:filename:0SaveV2_1/tensor_names:output:0"SaveV2_1/shape_and_slices:output:0savev2_1_const^SaveV2"/device:CPU:0*
_output_shapes
 *
dtypes
2Й
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0ShardedFilename_1:filename:0^SaveV2	^SaveV2_1"/device:CPU:0*
T0*
N*
_output_shapes
:
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix	^SaveV2_1"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: s

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints^SaveV2	^SaveV2_1*
T0*
_output_shapes
: "!

identity_1Identity_1:output:0*G
_input_shapes6
4: :
::	
:
: : : : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints2
SaveV2SaveV22
SaveV2_1SaveV2_1: : : : :	 :
 : :+ '
%
_user_specified_namefile_prefix: : : : 
З
`
'__inference_dropout_layer_call_fn_29575

inputs
identityЂStatefulPartitionedCallЈ
StatefulPartitionedCallStatefulPartitionedCallinputs*(
_output_shapes
:џџџџџџџџџ*
Tin
2*,
_gradient_op_typePartitionedCall-29322*K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_29311*
Tout
2**
config_proto

GPU 

CPU2J 8
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*'
_input_shapes
:џџџџџџџџџ22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
ѕ
^
B__inference_flatten_layer_call_and_return_conditional_losses_29250

inputs
identity^
Reshape/shapeConst*
dtype0*
_output_shapes
:*
valueB"џџџџ  e
ReshapeReshapeinputsReshape/shape:output:0*
T0*(
_output_shapes
:џџџџџџџџџY
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0**
_input_shapes
:џџџџџџџџџ:& "
 
_user_specified_nameinputs
е	
л
B__inference_dense_1_layer_call_and_return_conditional_losses_29346

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpЃ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
:	
*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*'
_output_shapes
:џџџџџџџџџ
*
T0 
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:
v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*'
_output_shapes
:џџџџџџџџџ
*
T0V
SoftmaxSoftmaxBiasAdd:output:0*'
_output_shapes
:џџџџџџџџџ
*
T0
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*'
_output_shapes
:џџџџџџџџџ
*
T0"
identityIdentity:output:0*/
_input_shapes
:џџџџџџџџџ::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp:& "
 
_user_specified_nameinputs: : 
д	
й
@__inference_dense_layer_call_and_return_conditional_losses_29538

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpЄ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0* 
_output_shapes
:
j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџЁ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*(
_output_shapes
:џџџџџџџџџ*
T0Q
ReluReluBiasAdd:output:0*(
_output_shapes
:џџџџџџџџџ*
T0
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*/
_input_shapes
:џџџџџџџџџ::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp:& "
 
_user_specified_nameinputs: : 
й
й
E__inference_sequential_layer_call_and_return_conditional_losses_29378
flatten_input(
$dense_statefulpartitionedcall_args_1(
$dense_statefulpartitionedcall_args_2*
&dense_1_statefulpartitionedcall_args_1*
&dense_1_statefulpartitionedcall_args_2
identityЂdense/StatefulPartitionedCallЂdense_1/StatefulPartitionedCallЇ
flatten/PartitionedCallPartitionedCallflatten_input**
config_proto

GPU 

CPU2J 8*(
_output_shapes
:џџџџџџџџџ*
Tin
2*,
_gradient_op_typePartitionedCall-29256*K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_29250*
Tout
2
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0$dense_statefulpartitionedcall_args_1$dense_statefulpartitionedcall_args_2**
config_proto

GPU 

CPU2J 8*(
_output_shapes
:џџџџџџџџџ*
Tin
2*,
_gradient_op_typePartitionedCall-29280*I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_29274*
Tout
2Р
dropout/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*,
_gradient_op_typePartitionedCall-29330*K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_29318*
Tout
2**
config_proto

GPU 

CPU2J 8*(
_output_shapes
:џџџџџџџџџ*
Tin
2
dense_1/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0&dense_1_statefulpartitionedcall_args_1&dense_1_statefulpartitionedcall_args_2**
config_proto

GPU 

CPU2J 8*
Tin
2*'
_output_shapes
:џџџџџџџџџ
*,
_gradient_op_typePartitionedCall-29352*K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_29346*
Tout
2В
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ
"
identityIdentity:output:0*:
_input_shapes)
':џџџџџџџџџ::::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall: : :- )
'
_user_specified_nameflatten_input: : 

њ
*__inference_sequential_layer_call_fn_29425
flatten_input"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identityЂStatefulPartitionedCallЕ
StatefulPartitionedCallStatefulPartitionedCallflatten_inputstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*
Tin	
2*'
_output_shapes
:џџџџџџџџџ
*,
_gradient_op_typePartitionedCall-29418*N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_29417*
Tout
2**
config_proto

GPU 

CPU2J 8
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:џџџџџџџџџ
"
identityIdentity:output:0*:
_input_shapes)
':џџџџџџџџџ::::22
StatefulPartitionedCallStatefulPartitionedCall:- )
'
_user_specified_nameflatten_input: : : : 
д
Ј
'__inference_dense_1_layer_call_fn_29598

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identityЂStatefulPartitionedCallщ
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tout
2**
config_proto

GPU 

CPU2J 8*'
_output_shapes
:џџџџџџџџџ
*
Tin
2*,
_gradient_op_typePartitionedCall-29352*K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_29346
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*'
_output_shapes
:џџџџџџџџџ
*
T0"
identityIdentity:output:0*/
_input_shapes
:џџџџџџџџџ::22
StatefulPartitionedCallStatefulPartitionedCall: :& "
 
_user_specified_nameinputs: "wL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*К
serving_defaultІ
K
flatten_input:
serving_default_flatten_input:0џџџџџџџџџ;
dense_10
StatefulPartitionedCall:0џџџџџџџџџ
tensorflow/serving/predict*>
__saved_model_init_op%#
__saved_model_init_op

NoOp:

layer-0
layer-1
layer_with_weights-0
layer-2
layer-3
layer_with_weights-1
layer-4
	optimizer
regularization_losses
trainable_variables
		variables

	keras_api

signatures
*L&call_and_return_all_conditional_losses
M__call__
N_default_save_signature"з
_tf_keras_sequentialИ{"class_name": "Sequential", "name": "sequential", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "sequential", "layers": [{"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "batch_input_shape": [null, 28, 28], "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 10, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}, "keras_version": "2.2.4-tf", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential", "layers": [{"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "batch_input_shape": [null, 28, 28], "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 10, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}, "training_config": {"loss": "sparse_categorical_crossentropy", "metrics": ["accuracy"], "weighted_metrics": null, "sample_weight_mode": null, "loss_weights": null, "optimizer_config": {"class_name": "SGD", "config": {"name": "SGD", "learning_rate": 0.009999999776482582, "decay": 0.0, "momentum": 0.0, "nesterov": false}}}}
З
regularization_losses
trainable_variables
	variables
	keras_api
*O&call_and_return_all_conditional_losses
P__call__"Ј
_tf_keras_layer{"class_name": "InputLayer", "name": "flatten_input", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": [null, 28, 28], "config": {"batch_input_shape": [null, 28, 28], "dtype": "float32", "sparse": false, "name": "flatten_input"}}
л
regularization_losses
trainable_variables
	variables
	keras_api
*Q&call_and_return_all_conditional_losses
R__call__"Ь
_tf_keras_layerВ{"class_name": "Flatten", "name": "flatten", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": [null, 28, 28], "config": {"name": "flatten", "trainable": true, "batch_input_shape": [null, 28, 28], "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
я

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
*S&call_and_return_all_conditional_losses
T__call__"Ъ
_tf_keras_layerА{"class_name": "Dense", "name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 784}}}}
Ћ
regularization_losses
trainable_variables
	variables
	keras_api
*U&call_and_return_all_conditional_losses
V__call__"
_tf_keras_layer{"class_name": "Dropout", "name": "dropout", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}
ѕ

kernel
bias
 regularization_losses
!trainable_variables
"	variables
#	keras_api
*W&call_and_return_all_conditional_losses
X__call__"а
_tf_keras_layerЖ{"class_name": "Dense", "name": "dense_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 10, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}}}
I
$iter
	%decay
&learning_rate
'momentum"
	optimizer
 "
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
З
(layer_regularization_losses
regularization_losses
)metrics

*layers
trainable_variables
+non_trainable_variables
		variables
M__call__
N_default_save_signature
*L&call_and_return_all_conditional_losses
&L"call_and_return_conditional_losses"
_generic_user_object
,
Yserving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper

,layer_regularization_losses
regularization_losses
-metrics

.layers
trainable_variables
/non_trainable_variables
	variables
P__call__
*O&call_and_return_all_conditional_losses
&O"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper

0layer_regularization_losses
regularization_losses
1metrics

2layers
trainable_variables
3non_trainable_variables
	variables
R__call__
*Q&call_and_return_all_conditional_losses
&Q"call_and_return_conditional_losses"
_generic_user_object
 :
2dense/kernel
:2
dense/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper

4layer_regularization_losses
regularization_losses
5metrics

6layers
trainable_variables
7non_trainable_variables
	variables
T__call__
*S&call_and_return_all_conditional_losses
&S"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper

8layer_regularization_losses
regularization_losses
9metrics

:layers
trainable_variables
;non_trainable_variables
	variables
V__call__
*U&call_and_return_all_conditional_losses
&U"call_and_return_conditional_losses"
_generic_user_object
!:	
2dense_1/kernel
:
2dense_1/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper

<layer_regularization_losses
 regularization_losses
=metrics

>layers
!trainable_variables
?non_trainable_variables
"	variables
X__call__
*W&call_and_return_all_conditional_losses
&W"call_and_return_conditional_losses"
_generic_user_object
:	 (2SGD/iter
: (2	SGD/decay
: (2SGD/learning_rate
: (2SGD/momentum
 "
trackable_list_wrapper
'
@0"
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper

	Atotal
	Bcount
C
_fn_kwargs
Dregularization_losses
Etrainable_variables
F	variables
G	keras_api
*Z&call_and_return_all_conditional_losses
[__call__"х
_tf_keras_layerЫ{"class_name": "MeanMetricWrapper", "name": "accuracy", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "accuracy", "dtype": "float32"}}
:  (2total
:  (2count
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
A0
B1"
trackable_list_wrapper

Hlayer_regularization_losses
Dregularization_losses
Imetrics

Jlayers
Etrainable_variables
Knon_trainable_variables
F	variables
[__call__
*Z&call_and_return_all_conditional_losses
&Z"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
A0
B1"
trackable_list_wrapper
т2п
E__inference_sequential_layer_call_and_return_conditional_losses_29378
E__inference_sequential_layer_call_and_return_conditional_losses_29477
E__inference_sequential_layer_call_and_return_conditional_losses_29498
E__inference_sequential_layer_call_and_return_conditional_losses_29364Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
і2ѓ
*__inference_sequential_layer_call_fn_29401
*__inference_sequential_layer_call_fn_29507
*__inference_sequential_layer_call_fn_29425
*__inference_sequential_layer_call_fn_29516Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
ш2х
 __inference__wrapped_model_29238Р
В
FullArgSpec
args 
varargsjargs
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *0Ђ-
+(
flatten_inputџџџџџџџџџ
Ь2ЩЦ
НВЙ
FullArgSpec
args
jself
jinputs
varargs
 
varkwjkwargs
defaults 

kwonlyargs

jtraining%
kwonlydefaultsЊ

trainingp 
annotationsЊ *
 
Ь2ЩЦ
НВЙ
FullArgSpec
args
jself
jinputs
varargs
 
varkwjkwargs
defaults 

kwonlyargs

jtraining%
kwonlydefaultsЊ

trainingp 
annotationsЊ *
 
ь2щ
B__inference_flatten_layer_call_and_return_conditional_losses_29522Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
б2Ю
'__inference_flatten_layer_call_fn_29527Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ъ2ч
@__inference_dense_layer_call_and_return_conditional_losses_29538Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Я2Ь
%__inference_dense_layer_call_fn_29545Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Т2П
B__inference_dropout_layer_call_and_return_conditional_losses_29570
B__inference_dropout_layer_call_and_return_conditional_losses_29565Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
2
'__inference_dropout_layer_call_fn_29580
'__inference_dropout_layer_call_fn_29575Д
ЋВЇ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
ь2щ
B__inference_dense_1_layer_call_and_return_conditional_losses_29591Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
б2Ю
'__inference_dense_1_layer_call_fn_29598Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
8B6
#__inference_signature_wrapper_29439flatten_input
Ь2ЩЦ
НВЙ
FullArgSpec
args
jself
jinputs
varargs
 
varkwjkwargs
defaults 

kwonlyargs

jtraining%
kwonlydefaultsЊ

trainingp 
annotationsЊ *
 
Ь2ЩЦ
НВЙ
FullArgSpec
args
jself
jinputs
varargs
 
varkwjkwargs
defaults 

kwonlyargs

jtraining%
kwonlydefaultsЊ

trainingp 
annotationsЊ *
 
*__inference_sequential_layer_call_fn_29516];Ђ8
1Ђ.
$!
inputsџџџџџџџџџ
p 

 
Њ "џџџџџџџџџ
Ѓ
B__inference_dense_1_layer_call_and_return_conditional_losses_29591]0Ђ-
&Ђ#
!
inputsџџџџџџџџџ
Њ "%Ђ"

0џџџџџџџџџ

 {
'__inference_dense_1_layer_call_fn_29598P0Ђ-
&Ђ#
!
inputsџџџџџџџџџ
Њ "џџџџџџџџџ
К
E__inference_sequential_layer_call_and_return_conditional_losses_29364qBЂ?
8Ђ5
+(
flatten_inputџџџџџџџџџ
p

 
Њ "%Ђ"

0џџџџџџџџџ

 
*__inference_sequential_layer_call_fn_29425dBЂ?
8Ђ5
+(
flatten_inputџџџџџџџџџ
p 

 
Њ "џџџџџџџџџ
Г
E__inference_sequential_layer_call_and_return_conditional_losses_29477j;Ђ8
1Ђ.
$!
inputsџџџџџџџџџ
p

 
Њ "%Ђ"

0џџџџџџџџџ

 {
'__inference_flatten_layer_call_fn_29527P3Ђ0
)Ђ&
$!
inputsџџџџџџџџџ
Њ "џџџџџџџџџЂ
@__inference_dense_layer_call_and_return_conditional_losses_29538^0Ђ-
&Ђ#
!
inputsџџџџџџџџџ
Њ "&Ђ#

0џџџџџџџџџ
 Ѓ
B__inference_flatten_layer_call_and_return_conditional_losses_29522]3Ђ0
)Ђ&
$!
inputsџџџџџџџџџ
Њ "&Ђ#

0џџџџџџџџџ
 Ў
#__inference_signature_wrapper_29439KЂH
Ђ 
AЊ>
<
flatten_input+(
flatten_inputџџџџџџџџџ"1Њ.
,
dense_1!
dense_1џџџџџџџџџ
К
E__inference_sequential_layer_call_and_return_conditional_losses_29378qBЂ?
8Ђ5
+(
flatten_inputџџџџџџџџџ
p 

 
Њ "%Ђ"

0џџџџџџџџџ

 |
'__inference_dropout_layer_call_fn_29580Q4Ђ1
*Ђ'
!
inputsџџџџџџџџџ
p 
Њ "џџџџџџџџџ|
'__inference_dropout_layer_call_fn_29575Q4Ђ1
*Ђ'
!
inputsџџџџџџџџџ
p
Њ "џџџџџџџџџ
 __inference__wrapped_model_29238u:Ђ7
0Ђ-
+(
flatten_inputџџџџџџџџџ
Њ "1Њ.
,
dense_1!
dense_1џџџџџџџџџ
Г
E__inference_sequential_layer_call_and_return_conditional_losses_29498j;Ђ8
1Ђ.
$!
inputsџџџџџџџџџ
p 

 
Њ "%Ђ"

0џџџџџџџџџ

 z
%__inference_dense_layer_call_fn_29545Q0Ђ-
&Ђ#
!
inputsџџџџџџџџџ
Њ "џџџџџџџџџЄ
B__inference_dropout_layer_call_and_return_conditional_losses_29570^4Ђ1
*Ђ'
!
inputsџџџџџџџџџ
p 
Њ "&Ђ#

0џџџџџџџџџ
 Є
B__inference_dropout_layer_call_and_return_conditional_losses_29565^4Ђ1
*Ђ'
!
inputsџџџџџџџџџ
p
Њ "&Ђ#

0џџџџџџџџџ
 
*__inference_sequential_layer_call_fn_29507];Ђ8
1Ђ.
$!
inputsџџџџџџџџџ
p

 
Њ "џџџџџџџџџ

*__inference_sequential_layer_call_fn_29401dBЂ?
8Ђ5
+(
flatten_inputџџџџџџџџџ
p

 
Њ "џџџџџџџџџ
