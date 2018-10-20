package gp.source;

import gp.verification.Verifier;
import org.opencv.core.Core;

public class Main {    
    public static void main(String[] args) {
        System.out.println(System.getProperty("java.library.path"));
        System.loadLibrary( Core.NATIVE_LIBRARY_NAME );   
                
        Verifier verifier = new Verifier();

//        verifier.testImgNameGenerator();
        verifier.testVerification();
//        System.out.println(verifier.verifySamePerson("Selena 2.jpg", "Selena 6.jpg"));

    }
}