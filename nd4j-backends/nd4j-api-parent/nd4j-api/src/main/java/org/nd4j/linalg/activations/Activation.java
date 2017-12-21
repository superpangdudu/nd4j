package org.nd4j.linalg.activations;

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.activations.impl.*;

/**
 * This enum is the factory for the activation function.
 *
 * Created by susaneraly on 12/8/16.
 */
public enum Activation {
    CUBE, ELU, HARDSIGMOID, HARDTANH, IDENTITY, LEAKYRELU, RATIONALTANH, RELU, RRELU, SIGMOID, SOFTMAX, SOFTPLUS, SOFTSIGN, TANH, RECTIFIEDTANH, SELU, SWISH;

    /**
     * Creates an instance of the activation function
     *
     * @return an instance of the activation function
     */
    public IActivation getActivationFunction() {
        switch (this) {
            case CUBE:
                return new ActivationCube();
            case ELU:
                return new ActivationELU();
            case HARDSIGMOID:
                return new ActivationHardSigmoid();
            case HARDTANH:
                return new ActivationHardTanH();
            case IDENTITY:
                return new ActivationIdentity();
            case LEAKYRELU:
                return new ActivationLReLU();
            case RATIONALTANH:
                return new ActivationRationalTanh();
            case RECTIFIEDTANH:
                return new ActivationRectifiedTanh();
            case RELU:
                return new ActivationReLU();
            case SELU:
                return new ActivationSELU();
            case SWISH:
                return new ActivationSwish();
            case RRELU:
                return new ActivationRReLU();
            case SIGMOID:
                return new ActivationSigmoid();
            case SOFTMAX:
                return new ActivationSoftmax();
            case SOFTPLUS:
                return new ActivationSoftPlus();
            case SOFTSIGN:
                return new ActivationSoftSign();
            case TANH:
                return new ActivationTanH();
            default:
                throw new UnsupportedOperationException("Unknown or not supported activation function: " + this);
        }
    }

    /**
     * Returns the activation function enum value
     *
     * @param name the case-insensitive opName of the activation function
     * @return the activation function enum value
     */
    public static Activation fromString(String name) {
        return Activation.valueOf(name.toUpperCase());
    }

    public SDVariable asSameDiff(SameDiff sd, SDVariable input) {
        return asSameDiff(null, sd, input);
    }

    public SDVariable asSameDiff(String variableName, SameDiff sd, SDVariable input){
        switch (this){
            case CUBE:
                return sd.pow(variableName, input, 3.0);
            case ELU:
                return sd.elu(variableName, input);
            case HARDTANH:
                return sd.hardTanh(variableName, input);
            case IDENTITY:
                return input;    //TODO Is this OK in all cases?
            case LEAKYRELU:
                return sd.leakyRelu(variableName, input, 0.0);
            case RELU:
                return sd.relu(variableName, input, 0.0);
            case SIGMOID:
                return sd.sigmoid(variableName, input);
            case SOFTMAX:
                return sd.softmax(variableName, input);
            case SOFTPLUS:
                return sd.softplus(variableName, input);
            case SOFTSIGN:
                return sd.softsign(variableName, input);
            case TANH:
                return sd.tanh(variableName, input);
            case HARDSIGMOID:
            case RATIONALTANH:
            case RRELU:
            case RECTIFIEDTANH:
            case SELU:
            case SWISH:
            default:
                throw new UnsupportedOperationException("Activation function not yet supported: " + this);
        }
    }

}
