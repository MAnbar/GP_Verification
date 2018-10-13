package gp.verification;

import gp.services.FileManager;
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
    static public final String Input_Images_PATH="inputs";
    static public final String Output_Images_PATH="outputs";
    
    private final ArrayList<String> classifiers;
    private final ArrayList<String> imgs;
    private final CascadeClassifier[] classifierArray;
    private final Facemark fm1;

    int counter;
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
    public Mat compare(Mat img1, Mat img2){
        
        Mat sImage1;
        Mat sImage2;
        
        MatOfRect faces1 = new MatOfRect();
        MatOfRect faces2 = new MatOfRect();

        classifierArray[0].detectMultiScale(img1, faces1);
        classifierArray[0].detectMultiScale(img2, faces2);

        if(faces1.empty()||faces2.empty()){
            return null;
        }
        ArrayList<MatOfPoint2f> landmarks1= new ArrayList<>();
        ArrayList<MatOfPoint2f> landmarks2= new ArrayList<>();
        
        fm1.fit(img1, faces1, landmarks1);
        fm1.fit(img2, faces2, landmarks2);

        MatOfPoint2f lm1 = landmarks1.get(0);
        MatOfPoint2f lm2 = landmarks2.get(0);

        
        for(Integer h=0;h<lm1.rows();h++){
            double[] dp=lm1.get(h,0);
            Point p =new Point(dp[0], dp[1]);
            Imgproc.circle(img1, p, 2,new Scalar(0,255,255), 2);
            
            dp=lm2.get(h,0);
            p =new Point(dp[0], dp[1]);
            Imgproc.circle(img2, p, 2,new Scalar(0,255,255), 2);
        }

        Point left27i1=getLandMarkPoint(lm1, 27);
        Point right28i1=getLandMarkPoint(lm1, 28);
        
        Point left27i2=getLandMarkPoint(lm2, 27);
        Point right28i2=getLandMarkPoint(lm2, 28);

        double dY1=right28i1.x-left27i1.x;
        double dX1=right28i1.y-left27i1.y;
        double angle1=Math.toDegrees(Math.atan2(dY1, dX1));
        double length1= Math.sqrt(dX1*dX1+dY1*dY1);
        Point eyes1Center = new Point( (left27i1.x + right28i1.x) * 0.5f, (left27i1.y + right28i1.y) * 0.5f );
        System.out.println(angle1);
        
        double dY2=right28i2.x-left27i2.x;
        double dX2=right28i2.y-left27i2.y;
        double angle2=Math.toDegrees(Math.atan2(dY2, dX2));
        double length2= Math.sqrt(dX1*dX1+dY1*dY1);
        Point eyes2Center = new Point( (left27i2.x + right28i2.x) * 0.5f, (left27i2.y + right28i2.y) * 0.5f );
        System.out.println(angle2);
        
        double scale=length2/length1;
//        Mat rot_mat = Imgproc.getRotationMatrix2D(eyes1Center, angle1, scale);
//        Imgproc.warpAffine(img1, sImage1, rot_mat, img2.size());                 
//        Mat homography = Calib3d.findHomography(lm1, lm2);
//        Imgproc.warpPerspective(img1, sImage1, homography, img2.size());
        sImage1=rotateScaleImg(img1, -angle1, scale);
        sImage2=rotateScaleImg(img2, -angle2, 1);
        writeImage("SC1", sImage1);
        writeImage("SC2", sImage2);
        return sImage1;
    }

    public Point getLandMarkPoint(MatOfPoint2f lm, int PointID){
        double[] dp=lm.get(PointID,0);
        return new Point(dp[0], dp[1]);
    }

}

