package gp.verification;

import gp.services.*;
import java.util.ArrayList;
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
    static public final String Input_Images_PATH="C:\\Users\\ElMof\\Documents\\Graduation Project\\Images\\Input";
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
    public void printFeatureMatrixes(double[][] fMat1,double[][] fMat2){
        int size=fMat1[0].length;
        for(int i=0;i<size;i++){
            for(int j=0;j<size;j++){
                System.out.println("Feature("+i+","+j+")= "+fMat1[i][j]+"] vs ["+fMat2[i][j]);
            }
        }
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
    public Point getLandMarkPoint(MatOfPoint2f lm, int PointID){
        double[] dp=lm.get(PointID,0);
        return new Point(dp[0], dp[1]);
    }
    public double getLandMarkDistance(MatOfPoint2f lm, int Point1, int Point2, double scale){
        double[] dp1=lm.get(Point1,0);
        double[] dp2=lm.get(Point2,0);

        Point P1= new Point(dp1[0], dp1[1]);
        Point P2= new Point(dp2[0], dp2[1]);

        return(getScaledDistance(P1,P2,scale));
    }
    public double getLandMarkScale(MatOfPoint2f lm1, MatOfPoint2f lm2, int P1,int P2){
        Point P1i1=getLandMarkPoint(lm1, P1);
        Point P2i1=getLandMarkPoint(lm1, P2);
        
        Point P1i2=getLandMarkPoint(lm2, P1);
        Point P2i2=getLandMarkPoint(lm2, P2);        
        
        double dY1=P2i1.x-P1i1.x;
        double dX1=P2i1.y-P1i1.y;
        double length1= Math.sqrt(dX1*dX1+dY1*dY1);
        
        double dY2=P2i2.x-P1i2.x;
        double dX2=P2i2.y-P1i2.y;
        double length2= Math.sqrt(dX2*dX2+dY2*dY2);
        
        return (length2/length1);
    }
    public void drawLandMarks( MatOfPoint2f lm, Mat img){
        for(Integer h=0;h<lm.rows();h++){
            double[] dp=lm.get(h,0);
            Point p =new Point(dp[0], dp[1]);
            Imgproc.circle(img, p, 2,new Scalar(0,255,255), 2);
        }
    }
//--============================================================================
    public boolean verifySamePerson(String img1Filename, String img2Filename){
        return RunPython.verifyFace(img1Filename, img2Filename);
    }
    public void testVerification(){
        
        String imgName1;
        String imgName2;
        boolean verifyResult;
        boolean actualResult;
        for(int i=0;i<imgs.size()-1;i++){
            imgName1=imgs.get(i);
            for(int j=i+1;j<imgs.size();j++){
                imgName2=imgs.get(j);
                StringManager.printPadded(imgName1, 15, false);
                System.out.print(" vs ");
                StringManager.printPadded(imgName2, 15, false);
                verifyResult=verifySamePerson(imgName1, imgName2);
                actualResult= imgName1.split(" ")[0].equals(imgName2.split(" ")[0]);
                StringManager.printPadded("Is "+actualResult, 9, false);
                System.out.println("Predicted "+verifyResult);
            }
        }
    }
    public void testImgNameGenerator(){
        String imgName1;
        String imgName2;
        boolean actualResult;
        for(int i=0;i<imgs.size()-1;i++){
            imgName1=imgs.get(i);
            for(int j=i+1;j<imgs.size();j++){
                imgName2=imgs.get(j);
                StringManager.printPadded(imgName1, 15, false);
                System.out.print(" vs ");
                StringManager.printPadded(imgName2, 15, false);
                actualResult= imgName1.split(" ")[0].equals(imgName2.split(" ")[0]);
                System.out.println(" Is "+actualResult+" ** "+imgName1.split(" ")[0]+" V "+imgName2.split(" ")[0]);
            }
        }
    }
}

