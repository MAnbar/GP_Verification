package gp.verification;

import gp.services.FileManager;
import java.util.ArrayList;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.face.Face;
import org.opencv.face.Facemark;
import org.opencv.highgui.HighGui;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;

public class Verifier {
    static public final String MODELS_PATH="resources/models";
    static public final String CLASSIFIERS_PATH="resources/classifiers";
    static public final String Input_Images_PATH="inputs";
    static public final String Output_Images_PATH="outputs";
    
    private final ArrayList<String> classifiers;
    private final ArrayList<String> imgs;
    private CascadeClassifier[] classifierArray;
    private Facemark fm1;

        
    public Verifier() {
        this.classifiers = FileManager.checkPath(CLASSIFIERS_PATH);
        this.imgs = FileManager.checkPath(Input_Images_PATH);
        
        this.classifierArray=new CascadeClassifier[classifiers.size()];

        
        for(int i=0;i<classifierArray.length;i++){
            classifierArray[i]= new CascadeClassifier(CLASSIFIERS_PATH+"/"+classifiers.get(i));
        }
        
        System.out.println(System.getProperty("java.library.path"));
        System.loadLibrary( Core.NATIVE_LIBRARY_NAME );   
        this.fm1 = Face.createFacemarkKazemi();
        fm1.loadModel(MODELS_PATH+"/face_landmark_model.dat");

    }
    
    public void showImage(Mat img){
        HighGui.namedWindow("image", HighGui.WINDOW_AUTOSIZE);
        HighGui.imshow("image", img);
        HighGui.waitKey();
    }
    
    public void Rotate(Mat img1,Mat img2){
        
    }
    
    public void detectAllImgLandmarks(){
                String imgName="";
        for(int n=0;n<imgs.size();n++){
            imgName=imgs.get(n).split("\\.")[0];
        /////////////////////////////////////////////////////////////////////////////////////////////////////////////
            for(int i=0;i<classifiers.size();i++){
            Mat origImg2 = Imgcodecs.imread(Input_Images_PATH+"/"+imgs.get(n));
            MatOfRect faces = new MatOfRect();

            classifierArray[i].detectMultiScale(origImg2, faces);

            if(faces.empty()){
                continue;
            }
            Facemark fm1= Face.createFacemarkKazemi();
            fm1.loadModel(MODELS_PATH+"/face_landmark_model.dat");
            ArrayList<MatOfPoint2f> landmarks= new ArrayList<>();
                fm1.fit(origImg2, faces, landmarks);

                for(int z=0;z<landmarks.size();z++){
                    MatOfPoint2f lm = landmarks.get(z);
                    
                    for(Integer h=0;h<lm.rows();h++){
                        double[] dp=lm.get(h,0);
                        Point p =new Point(dp[0], dp[1]);
                        Imgproc.circle(origImg2, p, 2,new Scalar(0,255,255), 2);
//                        p.y=p.y+8;
//                        Imgproc.putText(origImg2, h.toString(), p, 1, 1, new Scalar(0,255,255));
                    }
                }
                Imgcodecs.imwrite(Output_Images_PATH+"/Test"+imgName+classifiers.get(i)+".jpg", origImg2);
    //            showImage(origImg2);
            }
        /////////////////////////////////////////////////////////////////////////////
        }
    }
    
    public void detectSingleImgLandmarks(String imgName){
        
        for(int i=0;i<classifiers.size();i++){
        Mat origImg2 = Imgcodecs.imread(Input_Images_PATH+"/"+imgName);
        MatOfRect faces = new MatOfRect();

        classifierArray[i].detectMultiScale(origImg2, faces);

        if(faces.empty()){
            continue;
        }
        Facemark fm1= Face.createFacemarkKazemi();
        fm1.loadModel(MODELS_PATH+"/face_landmark_model.dat");
        ArrayList<MatOfPoint2f> landmarks= new ArrayList<>();

        fm1.fit(origImg2, faces, landmarks);

        for(int z=0;z<landmarks.size();z++){
            MatOfPoint2f lm = landmarks.get(z);

            for(Integer h=0;h<lm.rows();h++){
                double[] dp=lm.get(h,0);
                Point p =new Point(dp[0], dp[1]);
                Imgproc.circle(origImg2, p, 2,new Scalar(0,255,255), 2);
//              p.y=p.y+8;
//              Imgproc.putText(origImg2, h.toString(), p, 1, 1, new Scalar(0,255,255));
            }
        }
        Imgcodecs.imwrite(Output_Images_PATH+"/Test"+classifiers.get(i)+imgName, origImg2);
//          showImage(origImg2);
        }
    }
}

