package org.nd4j.linalg.api.ops.impl.layers.convolution;

import lombok.Builder;
import lombok.Getter;
import lombok.extern.slf4j.Slf4j;
import lombok.val;
import onnx.OnnxProto3;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.imports.graphmapper.tf.TFGraphMapper;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.Conv2DConfig;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.nd4j.linalg.util.ArrayUtil;
import org.tensorflow.framework.AttrValue;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.framework.NodeDef;

import java.util.*;


/**
 * Conv2D operation
 */
@Slf4j
@Getter
public class Conv2D extends DynamicCustomOp {

    protected  Conv2DConfig conv2DConfig;

    @Builder(builderMethodName = "builder")
    public Conv2D(SameDiff sameDiff,
                  SDVariable[] inputFunctions,
                  INDArray[] inputArrays, INDArray[] outputs,
                  Conv2DConfig conv2DConfig) {
        super(null,inputArrays,outputs);
        this.sameDiff = sameDiff;
        this.conv2DConfig = conv2DConfig;
        addArgs();
        sameDiff.putFunctionForId(this.getInstanceId(),this);;    //Normally called in DynamicCustomOp constructor, via setInstanceId - but sameDiff field is null at that point
        sameDiff.addArgsFor(inputFunctions, this);
    }

    public Conv2D() {}

    protected void addArgs() {
        addIArgument(new int[]{conv2DConfig.getKh(),
                conv2DConfig.getKw(),
                conv2DConfig.getSy(),
                conv2DConfig.getSx(),
                conv2DConfig.getPh(),
                conv2DConfig.getPw(),
                conv2DConfig.getDh(),
                conv2DConfig.getDw(),
                ArrayUtil.fromBoolean(conv2DConfig.isSameMode())});

    }

    @Override
    public void initWithArrays(Map<String, INDArray> arrayMap, Object... extraArgs) {
        val var = sameDiff.getVariable(args()[1].getVarName());

        //place holder variable
        if (var.getArr() == null) {
            INDArray weights = arrayMap.get(var.getVarName());

            if(weights == null) {
                throw new ND4JIllegalStateException("Array for variable " + var.getVarName() + " was null!");
            }

            int[] permuteOrder = SameDiff.permuteDataFormatForSameDiff(conv2DConfig.getOriginalInputFormat(),true);
            weights = (weights.permute(permuteOrder).dup('c'));
            conv2DConfig.setKh(weights.size(2));
            conv2DConfig.setKw(weights.size(3));
            sameDiff.updateVariable(var.getVarName(), weights);
        }


        val func = args()[0];
        INDArray arr = sameDiff.getArrForVarName(func.getVarName());
        if(arr == null) {
            val var2 = sameDiff.getVariable(func.getVarName());
            arr = var2.storeAndAllocateNewArray();
        }
        else {
            int[] permuteOrder = SameDiff.permuteDataFormatForSameDiff(conv2DConfig.getOriginalInputFormat(),false);
            arr = arr.permute(permuteOrder).dup('c');
        }

        addInputArgument(arr);



    }

    @Override
    public void initFromTensorFlow(NodeDef nodeDef, SameDiff initWith, Map<String, AttrValue> attributesForNode, GraphDef graph) {
        val aStrides = nodeDef.getAttrOrThrow("strides");
        val tfStrides = aStrides.getList().getIList();
        val sY = tfStrides.get(1);
        val sX = tfStrides.get(2);

        val aPadding = nodeDef.getAttrOrDefault("padding", null);

        val paddingMode = aPadding.getS().toStringUtf8();
        int kY = 1;
        int kX = 1;
        val args = args();
        INDArray arr = sameDiff.getVariable(args[1].getVarName()).getArr();
        if(arr == null) {
            arr = TFGraphMapper.getInstance().getNDArrayFromTensor(nodeDef.getInput(0), nodeDef, graph);
        }



        /**
         * From the TF docs:
         * data_format: An optional string from: "NHWC", "NCHW". Defaults to "NHWC".
         Specify the data format of the input and output data.
         With the default format "NHWC", the data is stored in the order of: [batch, height, width, channels].
         Alternatively, the format could be "NCHW", the data storage order of: [batch, channels, height, width].
         */
        /**
         * N: Number of examples
         * C: Number of channels
         * H: Height
         * W: Width
         *
         * Dl4j uses: NCHW
         * Weights should be: out channels, in channels, height,width
         */

        val dataFormat = !attributesForNode.containsKey("data_format") ? "NHWC" : attributesForNode.get("data_format").getS().toStringUtf8();
        int[] ret = SameDiff.permuteDataFormatForSameDiff(dataFormat,true);

        arr = (arr.permute(ret).dup('c'));
        val  varForOp = initWith.getVariable(args[1].getVarName());
        initWith.associateArrayWithVariable(arr, varForOp);

        kY = arr.size(2);
        kX = arr.size(3);


        boolean isSameMode = paddingMode.equalsIgnoreCase("SAME");
        Conv2DConfig conv2DConfig = Conv2DConfig.builder()
                .kh(kY)
                .kw(kX)
                .originalInputFormat(dataFormat)
                .sx(sX.intValue())
                .sy(sY.intValue())
                .isSameMode(isSameMode)
                .build();
        this.conv2DConfig = conv2DConfig;
        addArgs();

    }

    @Override
    public void initFromOnnx(OnnxProto3.NodeProto node, SameDiff initWith, Map<String, OnnxProto3.AttributeProto> attributesForNode, OnnxProto3.GraphProto graph) {
        val autoPad = !attributesForNode.containsKey("auto_pad") ? "VALID" : attributesForNode.get("auto_pad").getS().toStringUtf8();
        val dilations = attributesForNode.get("dilations");
        val dilationY = dilations == null ? 1 : dilations.getIntsList().get(0).intValue();
        val dilationX = dilations == null ? 1 : dilations.getIntsList().get(1).intValue();
        val group = attributesForNode.get("group");

        val kernelShape = attributesForNode.get("kernel_shape");
        int kY = kernelShape.getIntsList().get(0).intValue();
        int kX = kernelShape.getIntsList().size() < 2 ? kY : kernelShape.getIntsList().get(1).intValue();

        val vertexId = args()[0];

        INDArray arr = vertexId.getArr();
        arr = (arr.permute(3, 2, 0, 1).dup('c'));
        initWith.associateArrayWithVariable(arr, vertexId);



        val strides = attributesForNode.get("strides");
        val sY = strides.getIntsList().get(0);
        val sX = strides.getIntsList().size() < 2 ? sY : strides.getIntsList().get(1);
        boolean isSameMode = autoPad
                .equalsIgnoreCase("SAME");
        Conv2DConfig conv2DConfig = Conv2DConfig.builder()
                .dh(dilationY)
                .dw(dilationX)
                .kh(kY)
                .kw(kX)
                .sx(sX.intValue())
                .sy(sY.intValue())
                .isSameMode(isSameMode)
                .build();
        this.conv2DConfig = conv2DConfig;
        addArgs();

        addOutputArgument(arr);
    }

    @Override
    public String opName() {
        return "conv2d";
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> f1) {
        List<SDVariable> ret = new ArrayList<>();
        List<DifferentialFunction> inputs = new ArrayList<>();
        inputs.addAll(Arrays.asList(args()));
        inputs.add(f1.get(0));
        Conv2DDerivative conv2DDerivative = Conv2DDerivative.derivativeBuilder()
                .conv2DConfig(conv2DConfig)
                .outputs(outputArguments())
                .inputFunctions(inputs.toArray(new SDVariable[inputs.size()]))
                .build();
        ret.addAll(Arrays.asList(conv2DDerivative.outputVariables()));
        return ret;
    }


    @Override
    public String onnxName() {
        return "Conv";
    }

    @Override
    public String tensorflowName() {
        return "Conv2D";
    }
}
