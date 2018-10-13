package gp.source;

import gp.verification.Verifier;
import org.opencv.core.Core;
import org.opencv.core.Mat;

public class Main {    
    public static void main(String[] args) {
        System.out.println(System.getProperty("java.library.path"));
        System.loadLibrary( Core.NATIVE_LIBRARY_NAME );   
                
        Verifier verifier = new Verifier();
        Mat src1 = verifier.readImage("E1.jpg");
        Mat src2 = verifier.readImage("F2.jpg");
        verifier.compare(src1, src2);
        verifier.detectAllImgLandmarks();
//        verifier.showImage(verifier.rotateScaleImg(src, 0,2));
//        verifier.showImage(src);
 
    }
}