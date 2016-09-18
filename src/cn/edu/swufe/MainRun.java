package cn.edu.swufe;

public class MainRun {

	public static void main(String[] args) {
		MatrixMain mtr = new MatrixMain();
		mtr.trainMethod(400, 400, 400);
		MatrixPredict mtrp = new MatrixPredict();
		mtrp.predictMethod(400, 400, 400);
	}

}
