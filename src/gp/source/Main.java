package gp.source;

import gp.verification.Verifier;
import org.opencv.core.Core;
import org.opencv.core.Mat;

public class Main {    
    public static void main(String[] args) {
        System.out.println(System.getProperty("java.library.path"));
        System.loadLibrary( Core.NATIVE_LIBRARY_NAME );   
                
        Verifier verifier = new Verifier();
        Mat src = verifier.readImage("E1.jpg");

        verifier.compare(src, src);
//        verifier.detectAllImgLandmarks();
        verifier.showImage(verifier.rotateScaleImg(src, 0,2));
        verifier.showImage(src);

        
        
    }
}