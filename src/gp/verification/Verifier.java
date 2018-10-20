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
import org.opencv.dnn.Dnn;

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
//Testing=======================================================================
double[][] truePositive;
double[][] falsePositive;
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
        
        //Testing=======================================================================
        truePositive=new double[68][68];
        falsePositive=new double[68][68];
        for(int i=0;i<68;i++){
            for(int j=0;j<68;j++){
                truePositive[i][j]=0;
                falsePositive[i][j]=0;
            }
        }
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
    public void testVerification(double tolerance){
        String imgName1="";
        String imgName2="";
        boolean result;
        int trueTotal=0;
        int falseTotal=0;
        boolean isTrue;
        Mat img1;
        Mat img2;
        for(int i=0;i<imgs.size();i++){
            imgName1=imgs.get(i).split("\\.")[0];
            img1 = Imgcodecs.imread(Input_Images_PATH+"/"+imgs.get(i));
            for(int j=i;j<imgs.size();j++){
                if(i==j){
                    continue;
                }
                System.out.println("======================================================================");
                imgName2=imgs.get(j).split("\\.")[0];
                img2 = Imgcodecs.imread(Input_Images_PATH+"/"+imgs.get(j));
                if(imgName1.charAt(0)==imgName2.charAt(0)){
                    isTrue=true;
                }
                else{
                    isTrue=false;
                }
                result=compare(img1, img2, tolerance,isTrue);
                if(result==isTrue){
                    trueTotal++;
                }
                else{
                    falseTotal++;
                }
                System.out.println(imgName1+" vs "+imgName2+" Same? : "+result);
            }
        }
        System.out.println("TruePositive Features Vs False Positive Features~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~");
        printFeatureMatrixes(truePositive, falsePositive);
        System.out.println("True= "+trueTotal);
        System.out.println("False = "+falseTotal);
    }
    public boolean compare(Mat img1, Mat img2,double tolerance, boolean isTrue){
        
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
        
//        drawLandMarks(lm1, img1);
//        drawLandMarks(lm2, img2);

        //39-42 Inner Eyes
        //36-45 Outer Eyes
        //21-22 Inner Eyebrows
        //17-26 Outer Eyebrows
        //48-54 Outer Mouth
        //31-35 Nose Width // 33 Nose Center
        //27-30 Nose Top
        //0to7-15to9 Face Outline
        //8 Chin
        double scale2Over1_2728=getLandMarkScale(lm1, lm2, 27, 28);
        double scale2Over1_2122=getLandMarkScale(lm1, lm2, 21, 22);
        double scale2Over1_3942=getLandMarkScale(lm1, lm2, 39, 42);
        double scale2Over1_3645=getLandMarkScale(lm1, lm2, 36, 45);
        double scale2Over1_3135=getLandMarkScale(lm1, lm2, 31, 35);
        double scale2Over1_4854=getLandMarkScale(lm1, lm2, 48, 54);
        double scale2Over1_0313=getLandMarkScale(lm1, lm2, 03, 13);
        
//        System.out.println("#########################################################");
//        System.out.println("Scale 2728:"+scale2Over1_2728);
//        System.out.println("Scale Inner EyeBrows (2122):"+scale2Over1_2122);
//        System.out.println("Scale Inner Eyes     (3942):"+scale2Over1_3942);
//        System.out.println("Scale Outer Eyes     (3645):"+scale2Over1_3645);
//        System.out.println("Scale Nose Width     (3135):"+scale2Over1_3135);
//        System.out.println("Scale Mouth Width    (4854):"+scale2Over1_4854);
//        System.out.println("Scale Face  Width    (0313):"+scale2Over1_0313);      
        double avgEyeScale=(scale2Over1_3942+scale2Over1_3645)/2;
//        double avgScales=(scale2Over1_2122+scale2Over1_3942+scale2Over1_3645+scale2Over1_3135+scale2Over1_4854+scale2Over1_0313)/6;
        double avgScales=(scale2Over1_3942+scale2Over1_3645+scale2Over1_3135)/3;
        System.out.println("#########################################################");
        System.out.println("Scale AVG Eyes (3942)(3645):"+avgEyeScale);        
        System.out.println("Average of All Scales      :"+avgScales);        
        System.out.println("#########################################################");
//        System.out.println("Eyes: "+verifyUsingWholeDifference(lm1,lm2, avgEyeScale,tolerance,isTrue));
        System.out.println("000000000000000000000000000000000000000000000000000000000");
//        System.out.println("All: "+verifyUsingWholeDifference(lm1,lm2, avgScales,tolerance,isTrue));
        System.out.println("000000000000000000000000000000000000000000000000000000000");
        
        
//      return  verifyUsingSelectiveDifference(lm1,lm2, avgScales,tolerance,isTrue);
      return  verifyUsingSelectiveDifference(lm1,lm2, avgEyeScale,tolerance,isTrue);

//        return verifyUsingWholeDifference(lm1,lm2, scale2Over1_3942,tolerance,isTrue);  
    }
    public boolean verifyUsingWholeDifference( MatOfPoint2f origLM1, MatOfPoint2f origLM2, double scale2Over1,double tolerance,boolean isTrue){
        max=0;
        
        double[][] featureLM1 = createDifferenceFeatureMatrix(origLM1,scale2Over1);
        double[][] featureLM2 = createDifferenceFeatureMatrix(origLM2,1);

        printFeatureMatrixes(featureLM1,featureLM2);

        boolean result = compareLandmarks(featureLM1,featureLM2,tolerance,isTrue);
        System.out.println("Max:"+max);
        return result;
    }
    public boolean verifyUsingSelectiveDifference( MatOfPoint2f origLM1, MatOfPoint2f origLM2, double scale2Over1,double tolerance,boolean isTrue){
        max=0;
        
        double[] featureLM1 = createSelectiveDifferenceFeatureMatrix(origLM1,scale2Over1);
        double[]featureLM2 = createSelectiveDifferenceFeatureMatrix(origLM2,1);

//        printFeatureMatrixes(featureLM1,featureLM2);

        boolean result = compareSelectiveLandmarks(featureLM1,featureLM2,tolerance,isTrue);
        System.out.println("Max:"+max);
        return result;
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
            }
        }
        return featureLM;
    }
    public double[] createSelectiveDifferenceFeatureMatrix( MatOfPoint2f origLM, double scale){
        
        //Eyes :36.39|42.45-39.42-36.45-39.27|27.42
        //Nose :31.35-27.30-27.33
        //Mouth:48.54
        //Face :0to7-16to9
        //Cross:8.39|8.42-8.36|8.45-8.31|8.35
        //     :0.36|45.16-39.31|42.35-36.33|33.45
//        int[]features1= {39,36,31,27,27,48,0 ,1 ,2 ,3 ,4 ,5 ,6 ,7};
//        int[]features2= {42,45,35,30,33,54,16,15,14,13,12,11,10,9};
//        int[]Avg1=      {39,36,8 ,8 ,8 ,0 ,39,36};
//        int[]Avg2=      {27,39,39,36,31,36,31,33};
//        int[]Avg3=      {27,42,8 ,8 ,8 ,45,42,33};
//        int[]Avg4=      {42,45,42,45,35,16,35,45};
        int[]features1= {39,36,31,27,27,39,42};
        int[]features2= {42,45,35,30,33,27,27};
        int[]Avg1=      {36,8 ,8 ,8 ,39,36};
        int[]Avg2=      {39,39,36,31,31,33};
        int[]Avg3=      {42,8 ,8 ,8 ,42,33};
        int[]Avg4=      {45,42,45,35,35,45};
        
        int sizeFM = features1.length;
        int sizeAM = Avg1.length;
        double[] featureLM = new double[sizeFM+sizeAM];

        Point p1;
        Point p2;

//        System.out.println("Distances:@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@");
        for(int i=0;i<sizeFM;i++){
            p1=getLandMarkPoint(origLM, features1[i]);
            p2=getLandMarkPoint(origLM, features2[i]);
            featureLM[i]=getScaledDistance(p1, p2, scale);
//            System.out.println("Distance("+features1[i]+","+features2[i]+")= "+featureLM[i]);
        }
        double A1;
        double A2;
        int Index;
        for(int i=sizeFM;i<sizeAM+sizeFM;i++){
            Index=i-sizeFM;
            p1=getLandMarkPoint(origLM, Avg1[Index]);
            p2=getLandMarkPoint(origLM, Avg2[Index]);
            A1=getScaledDistance(p1, p2, scale);
            p1=getLandMarkPoint(origLM, Avg3[Index]);
            p2=getLandMarkPoint(origLM, Avg4[Index]);
            A2=getScaledDistance(p1, p2, scale);
            featureLM[i]=(A1+A2)/2;
//            System.out.println("Distance("+Avg1[Index]+","+Avg2[Index]+"|"+Avg3[Index]+","+Avg4[Index]+")= "+featureLM[i]);
            i++;
        }
        return featureLM;
    }

    public boolean compareLandmarks(double[][] featureLM1,double[][] featureLM2,double tolerance, boolean isTrue){
        
        int sizeLM = featureLM1[0].length;
        if(sizeLM!=featureLM2[0].length){
            return false;
        }
        int sizeRows = featureLM1.length;
        
        double delta;
        double dAvg;
        double d1;
        double d2;
        double fault=0;
        for (int i=0; i<sizeRows;i++){
            for(int j=0;j<sizeLM;j++){
                d1=featureLM1[i][j];
                d2=featureLM2[i][j];
                delta=Math.abs(d1-d2);
                dAvg=(d1+d2)/2;
                if(dAvg>0){
                    if(delta/dAvg>max){
                        max=delta/dAvg;
                    }
                    if(delta/dAvg<tolerance){
                        if(isTrue){
                            truePositive[i][j]++;
                        }
                        else{
                            falsePositive[i][j]++;
                        }
                    }
                    fault+=delta/dAvg;
                }
//                if(max>tolerance){
//                    return false;
//                }
            }
        }
        fault=fault/(sizeLM*sizeRows);
        System.out.println("Fault="+fault);
        if(fault<tolerance){
            return true;
        }
        else{
            return false;
        }
    }
    public boolean compareSelectiveLandmarks(double[] featureLM1,double[] featureLM2,double tolerance, boolean isTrue){
        
        int sizeLM = featureLM1.length;
        if(sizeLM!=featureLM2.length){
            return false;
        }
        
        double delta;
        double dAvg;
        double d1;
        double d2;
        double fault=0;
        for (int i=0; i<sizeLM;i++){
                d1=featureLM1[i];
                d2=featureLM2[i];
                delta=Math.abs(d1-d2);
                dAvg=(d1+d2)/2;
//                dAvg=d2;
                if(dAvg>0){
                    if(delta/dAvg>max){
                        max=delta/dAvg;
                    }
                    if(delta/dAvg<tolerance){
                        if(isTrue){
                            truePositive[i][0]++;
                        }
                        else{
                            falsePositive[i][0]++;
                        }
                    }
                    fault+=delta/dAvg;
                }
//                if(max>tolerance){
//                    return false;
//                }
            
        }
        fault=fault/(sizeLM);
        System.out.println("Fault="+fault);
        if(fault<tolerance){
            return true;
        }
        else{
            return false;
        }
    }
}

