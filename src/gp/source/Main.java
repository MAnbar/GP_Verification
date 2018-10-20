package gp.source;

import gp.verification.Verifier;
import org.opencv.core.Core;
import org.opencv.core.Mat;

public class Main {    
    public static void main(String[] args) {
        System.out.println(System.getProperty("java.library.path"));
        System.loadLibrary( Core.NATIVE_LIBRARY_NAME );   
                
        Verifier verifier = new Verifier();
        Mat src1 = verifier.readImage("EM2.jpg");
        Mat src2 = verifier.readImage("EM1.jpg");
        Mat src3 = verifier.readImage("EM1.jpg");
        Mat src4 = verifier.readImage("M3.jpg");
        Mat src5 = verifier.readImage("S3.jpg");
        Mat src6 = verifier.readImage("A3.jpg");
        
//        Mat src7 = verifier.readImage("T1.jpg");
//        Mat src8 = verifier.readImage("T3.jpg");
//        Mat src9 = verifier.readImage("T4.jpg");
//        Mat src10 = verifier.readImage("T5.jpg");
//        Mat src11 = verifier.readImage("T8.jpg");

        System.out.println("True:"+verifier.compare(src2, src1, 0.03,true));
//        System.out.println("False:"+verifier.compare(src1, src4, 0.03,true));
//        System.out.println("False:"+verifier.compare(src5, src1, 0.03,true));
//        System.out.println("False:"+verifier.compare(src1, src6, 0.03,true));
        
//        System.out.println("True:"+verifier.compare(src2, src1, 0.03,true));
//        System.out.println("True:"+verifier.compare(src3, src1, 0.03,true));
//        System.out.println("True:"+verifier.compare(src3, src2, 0.03,true));
//        System.out.println("False:"+verifier.compare(src4, src1, 0.03,true));
//        System.out.println("False:"+verifier.compare(src5, src1, 0.03,true));
//        System.out.println("False:"+verifier.compare(src6, src1, 0.03,true));
//        verifier.testVerification(0.03);
//        System.out.println("Same Person:"+verifier.compare(src1, src2,0.1));
//        verifier.detectAllImgLandmarks();
//        verifier.showImage(verifier.rotateScaleImg(src, 0,2));
//        verifier.showImage(src); 
//==============================================================================
//verifier.testVerification(0.12);
//True Positive= 134
//False = 191

//verifier.testVerification(0.09);
//True Positive= 194
//False = 131
    }
}