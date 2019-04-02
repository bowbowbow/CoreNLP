/**
 *
 */
package edu.stanford.nlp.neural;

import static org.junit.Assert.*;

import java.util.Random;

import org.junit.Assert;
import org.ejml.simple.SimpleMatrix;
import org.junit.Test;

/**
 * @author Minh-Thang Luong <lmthang@stanford.edu>, created on Nov 15, 2013
 */
public class NeuralUtilsTest {

    @Test
    public void testCosine() {
        double[][] values = new double[1][5];
        values[0] = new double[]{0.1, 0.2, 0.3, 0.4, 0.5};
        SimpleMatrix vector1 = new SimpleMatrix(values);

        values[0] = new double[]{0.5, 0.4, 0.3, 0.2, 0.1};
        SimpleMatrix vector2 = new SimpleMatrix(values);

        assertEquals(0.35000000000000003, NeuralUtils.dot(vector1, vector2), 1e-5);
        assertEquals(0.6363636363636364, NeuralUtils.cosine(vector1, vector2), 1e-5);

        vector1 = vector1.transpose();
        vector2 = vector2.transpose();
        assertEquals(0.35000000000000003, NeuralUtils.dot(vector1, vector2), 1e-5);
        assertEquals(0.6363636363636364, NeuralUtils.cosine(vector1, vector2), 1e-5);
    }

    public void testIsZero() {
        double[][] values = new double[][]{{0.1, 0.2, 0.3, 0.4, 0.5}, {0.0, 0.0, 0.0, 0.0, 0.0}};
        SimpleMatrix vector1 = new SimpleMatrix(values);
        assertFalse(NeuralUtils.isZero(vector1));

        values = new double[][]{{0.0, 0.0, 0.0, 0.0, 0.0}, {0.0, 0.0, 0.0, 0.0, 0.0}};
        vector1 = new SimpleMatrix(values);
        assertTrue(NeuralUtils.isZero(vector1));
    }

    @Test
    public void testSimpleTensor() {
        SimpleTensor tensor1 = new SimpleTensor(2, 5, 2);
        Assert.assertEquals(tensor1.numRows(), 2);
        Assert.assertEquals(tensor1.numCols(), 5);
        Assert.assertEquals(tensor1.numSlices(), 2);
        Assert.assertEquals(tensor1.getNumElements(), 2 * 5 * 2);

        SimpleTensor tensor2 = SimpleTensor.random(2, 5, 2, 1.0, 2.0, new Random());
        Assert.assertEquals(tensor2.getNumElements(), 2 * 5 * 2);
        SimpleMatrix tensor2Vector0 = tensor2.getSlice(0);
        SimpleMatrix tensor2Vector1 = tensor2.getSlice(1);
        Assert.assertEquals(tensor2.elementSum(), tensor2Vector0.elementSum() + tensor2Vector1.elementSum(), 1e-6);
        Assert.assertTrue(tensor2.plus(tensor2.scale(-1.0)).isZero());

        tensor2.setSlice(0, tensor2Vector1);
        Assert.assertEquals(tensor2.elementSum(), tensor2Vector1.elementSum() * 2, 1e-6);

        double[][] values1 = new double[][]{{0.1, 0.2}, {0.3, 0.4}};
        SimpleMatrix vector1 = new SimpleMatrix(values1);
        double[][] values2 = new double[][]{{-0.1, -0.2}, {-0.3, -0.4}};
        SimpleMatrix vector2 = new SimpleMatrix(values2);

        SimpleMatrix[] slices = new SimpleMatrix[2];
        slices[0] = vector1;
        slices[1] = vector2;

        SimpleTensor tensor3 = new SimpleTensor(slices);
        Assert.assertEquals(tensor3.toString(), "Slice 0\nType = dense , numRows = 2 , numCols = 2\n 0.100   0.200  \n 0.300   0.400  \nSlice 1\nType = dense , numRows = 2 , numCols = 2\n-0.100  -0.200  \n-0.300  -0.400  \n");
        Assert.assertEquals(tensor3.toString("%f"), "Slice 0\n" +
                "Type = dense , numRows = 2 , numCols = 2\n" +
                "0.100000 0.200000 \n" +
                "0.300000 0.400000 \n" +
                "Slice 1\n" +
                "Type = dense , numRows = 2 , numCols = 2\n" +
                "-0.100000 -0.200000 \n" +
                "-0.300000 -0.400000 \n");
        Assert.assertEquals(tensor3.elementMult(tensor3).elementSum(), 0.6, 1e-6);

        //        System.out.println(tensor3.toString("%f"));
    }
}
