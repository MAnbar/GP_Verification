package gp.verification;

import gp.services.FileManager;
import java.util.ArrayList;
import java.util.Arrays;
import org.opencv.core.Core;
import static org.opencv.core.CvType.CV_64F;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
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
    private final CascadeClassifier[] classifierArray;
    private final Facemark fm1;

    int counter;
    double max;
//--============================================================================
    public Verifier() {
        this.counter = 0;
        
        this.classifiers = FileManager.checkPath(CLASSIFIERS_PATH);
        this.imgs = FileManager.checkPath(Input_Images_PATH);
        
        this.classifierArray=new CascadeClassifier[classifiers.size()];
        for(int i=0;i<classifierArray.length;i++){
            classifierArray[i]= new CascadeClassifier(CLASSIFIERS_PATH+"/"+classifiers.get(i));
        }
        
        this.fm1 = Face.createFacemarkKazemi();
        fm1.loadModel(MODELS_PATH+"/face_landmark_model.dat");
    }
//--============================================================================
    public void showImage(Mat img){
        String windowName="image"+(++counter);
        HighGui.namedWindow(windowName, HighGui.WINDOW_AUTOSIZE);
        HighGui.imshow(windowName, img);
        HighGui.waitKey();
//        HighGui.destroyWindow(windowName);
    }
    
    public Mat readImage(String imgName){
        return Imgcodecs.imread(Input_Images_PATH+"/"+imgName);
    }
    public void writeImage(String imgName, Mat img){
        Imgcodecs.imwrite(Output_Images_PATH+"/Out"+imgName+".jpg", img);
    }
 //--============================================================================   
    public Mat rotateScaleImg(Mat src, double angle, double scale){
        Size size=new Size(src.cols(), src.rows());
        Mat dst = new Mat(size, src.type());
        
        Point center = new Point(src.cols()/2, src.rows()/2);
        Mat r = Imgproc.getRotationMatrix2D(center, angle, scale);
        Imgproc.warpAffine(src, dst, r, size);
        return dst;
    }
    
    public Mat resizeImg(Mat src, int cols, int rows){
        Mat dst = new Mat();
        Size size=new Size(cols, rows);
        Imgproc.resize(src, dst, size);
        return dst;
    }
    
//--============================================================================
    public void detectAllImgLandmarks(){
        String imgName="";
        for(int n=0;n<imgs.size();n++){
            imgName=imgs.get(n).split("\\.")[0];
            for(int i=0;i<classifiers.size();i++){
            Mat origImg2 = Imgcodecs.imread(Input_Images_PATH+"/"+imgs.get(n));
            MatOfRect faces = new MatOfRect();

            classifierArray[i].detectMultiScale(origImg2, faces);

            if(faces.empty()){
                continue;
            }
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
        }
    }
    
    public Mat detectSingleImgLandmarks(Mat src){
        
        Mat dst = src.clone();
        MatOfRect faces = new MatOfRect();

        classifierArray[0].detectMultiScale(dst, faces);
        if(faces.empty()){
            return null;
        }

        ArrayList<MatOfPoint2f> landmarks= new ArrayList<>();
        fm1.fit(dst, faces, landmarks);

        for(int z=0;z<landmarks.size();z++){
            MatOfPoint2f lm = landmarks.get(z);

            for(Integer h=0;h<lm.rows();h++){
                double[] dp=lm.get(h,0);
                Point p =new Point(dp[0], dp[1]);
                Imgproc.circle(dst, p, 2,new Scalar(0,255,255), 2);

            }
        }
        return dst;
    }
//--============================================================================
    public boolean compare(Mat img1, Mat img2,double tolerance){
        
        Mat fMat1;
        Mat fMat2;
        
        MatOfRect faces1 = new MatOfRect();
        MatOfRect faces2 = new MatOfRect();

        classifierArray[0].detectMultiScale(img1, faces1);
        classifierArray[0].detectMultiScale(img2, faces2);

        if(faces1.empty()||faces2.empty()){
            return false;
        }
        ArrayList<MatOfPoint2f> landmarks1= new ArrayList<>();
        ArrayList<MatOfPoint2f> landmarks2= new ArrayList<>();
        
        fm1.fit(img1, faces1, landmarks1);
        fm1.fit(img2, faces2, landmarks2);

        MatOfPoint2f lm1 = landmarks1.get(0);
        MatOfPoint2f lm2 = landmarks2.get(0);
        
//        System.out.println(lm1.dump());
//        System.out.println(lm2.dump());
//        System.out.println(lm1.get(0, 0).toString());
        
        drawLandMarks(lm1, img1);
        drawLandMarks(lm2, img2);

        Point left27i1=getLandMarkPoint(lm1, 27);
        Point right28i1=getLandMarkPoint(lm1, 28);
        
        Point left27i2=getLandMarkPoint(lm2, 27);
        Point right28i2=getLandMarkPoint(lm2, 28);

        double dY1=right28i1.x-left27i1.x;
        double dX1=right28i1.y-left27i1.y;
        double length1= Math.sqrt(dX1*dX1+dY1*dY1);
//        double angle1=Math.toDegrees(Math.atan2(dY1, dX1));
//        Point eyes1Center = new Point( (left27i1.x + right28i1.x) * 0.5f, (left27i1.y + right28i1.y) * 0.5f );
//        System.out.println(angle1);
        
        double dY2=right28i2.x-left27i2.x;
        double dX2=right28i2.y-left27i2.y;
        double length2= Math.sqrt(dX2*dX2+dY2*dY2);
//        double angle2=Math.toDegrees(Math.atan2(dY2, dX2));
//        Point eyes2Center = new Point( (left27i2.x + right28i2.x) * 0.5f, (left27i2.y + right28i2.y) * 0.5f );
//        System.out.println(angle2);
        
        double scale2To1=length2/length1;
        
        return verifyUsingWholeDifference(lm1,lm2, scale2To1,tolerance);        
    }

    public Point getLandMarkPoint(MatOfPoint2f lm, int PointID){
        double[] dp=lm.get(PointID,0);
        return new Point(dp[0], dp[1]);
    }

    public void drawLandMarks( MatOfPoint2f lm, Mat img){
        for(Integer h=0;h<lm.rows();h++){
            double[] dp=lm.get(h,0);
            Point p =new Point(dp[0], dp[1]);
            Imgproc.circle(img, p, 2,new Scalar(0,255,255), 2);
        }
    }
    
    public boolean verifyUsingWholeDifference( MatOfPoint2f origLM1, MatOfPoint2f origLM2, double scale2Over1,double tolerance){
        max=0;
        
        double[][] featureLM1 = createDifferenceFeatureMatrix(origLM1,scale2Over1);
        double[][] featureLM2 = createDifferenceFeatureMatrix(origLM2,1);

        
        boolean result = compareLandmarks(featureLM1,featureLM2,tolerance);
        System.out.println("Max:"+max);
        return result;
    }
    
    public double getLength(double dX, double dY){
        return Math.sqrt(dX*dX+dY*dY);
    }
    
    public Point subtractPoints(Point a, Point b, double scale){
        return new Point(Math.abs(a.x-b.x)*scale, Math.abs(a.y-b.y)*scale);
    }
    
    public double getScaledDistance(Point a,Point b,double scale){
        double dX=a.x-b.x;
        double dY=a.y-b.y;
        return Math.sqrt(dX*dX+dY*dY)*scale;
    }
    
    public double[][] createDifferenceFeatureMatrix( MatOfPoint2f origLM, double scale){
        
        int sizeLM = origLM.rows();
        double[][] featureLM = new double[sizeLM][sizeLM];

        double[] dp1;
        double[] dp2;
        Point pi;
        Point pj;

        for(int i=0;i<sizeLM;i++){
            dp1=origLM.get(i,0);
            for(int j=0;j<sizeLM;j++){
                dp2=origLM.get(j,0);
                pi = new Point(dp1[0], dp1[1]);
                pj = new Point(dp2[0], dp2[1]);
                featureLM[i][j]= getScaledDistance(pi, pj, scale);
                i=i;
            }
        }
        return featureLM;
    }
    
    public boolean compareLandmarks(double[][] featureLM1,double[][] featureLM2,double tolerance){
        
        int sizeLM = featureLM1[0].length;
        if(sizeLM!=featureLM2[0].length){
            return false;
        }
        
        double delta;
        double d1;
        double d2;
        for (int i=0; i<sizeLM;i++){
            for(int j=0;j<sizeLM;j++){
                d1=featureLM1[i][j];
                d2=featureLM2[i][j];
                delta=Math.abs(d1-d2);
                if(delta/d1>max){
                    max=delta/d1;
                }
                if(delta/d2>max){
                    max=delta/d2;
                }
                if(max>tolerance){
                    return false;
                }
            }
        }
        return true;
    }
    
    public void testVerification(double tolerance){
        String imgName1="";
        String imgName2="";
        Mat img1;
        Mat img2;
        for(int i=0;i<imgs.size();i++){
            imgName1=imgs.get(i).split("\\.")[0];
            img1 = Imgcodecs.imread(Input_Images_PATH+"/"+imgs.get(i));
            for(int j=0;j<imgs.size();j++){
                if(i==j){
                    continue;
                }
                imgName2=imgs.get(j).split("\\.")[0];
                img2 = Imgcodecs.imread(Input_Images_PATH+"/"+imgs.get(j));
                System.out.println(imgName1+" vs "+imgName2+" Same? : "+compare(img1, img2, tolerance));
            }
        }
    }

}

