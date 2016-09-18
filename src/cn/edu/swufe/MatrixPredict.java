package cn.edu.swufe;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;
import java.util.StringTokenizer;
import java.util.Vector;

import service.svm_predict;
import service.svm_train;
import libsvm.svm;
import libsvm.svm_model;
import libsvm.svm_node;
import libsvm.svm_parameter;
import libsvm.svm_problem;

public class MatrixPredict {
	private Vector<Double> pdthashflag = new Vector<Double>();
	private Vector<HashMap<Vector<Integer>, Double>> pdthashdata = new Vector<HashMap<Vector<Integer>, Double>>();	
	private double[] vx;
	private double[] vy;
	private double[] vz;
	private double beta;
	private double rho;
	private double b;
	private static double bata;
	private svm_model model;
	private svm_problem prob;
	private svm_parameter param;
	
	public static void main(String[] args) {
		//�ֶ�����bֵ,��ʽΪ(-rho)/beta
		//double b = -0.0011832847404535547/0.006231575848556862;
		MatrixPredict mtrp = new MatrixPredict();
		mtrp.predictMethod(100,2,2);
	}
	
	public void predictMethod(int x,int y,int z){		
		
		readHashData();		
		double bata = read_vector(x,y,z);
		read_rho();
		computer_b();
		predict(bata);		
	}
	
	private void computer_b(){
		b = (-rho)/beta;
		System.out.println("b��ֵ�ǣ�"+b);
	}
	
	private void read_rho(){
		BufferedReader fp;
		try{
			fp = new BufferedReader(new FileReader("..\\recsys_model.txt"));		
			for (int i = 1; i < 5; i++) {
				fp.readLine();
				}
			StringTokenizer st = new StringTokenizer(fp.readLine(), " \t\n\r\f:");
			st.nextToken();
			rho = Double.valueOf(st.nextToken()).doubleValue();
			}catch(IOException e){
				e.printStackTrace();
			}
	}
	
	private void readHashData() {
		try {
			BufferedReader fp;
			fp = new BufferedReader(new FileReader(
					"D:\\STR\\SampleTestTensorPrint.txt"));// �������ѵ�����ݣ��������ݣ��ĵ�ַ
			while (true) { // ���ݾ�������ĳ���ѭ��
				HashMap<Vector<Integer>, Double> datahashmap = new HashMap<Vector<Integer>, Double>();
				String line;
				line = fp.readLine();
				if (line == null)
					break;
				StringTokenizer st = new StringTokenizer(line, " \t\n\r\f");
				int length = st.countTokens();
				pdthashflag.add((Double) atof(st.nextToken()));
				for (int i = 1; i < length; i++) {
					Vector<Integer> tempt = new Vector<Integer>();
					StringTokenizer feature = new StringTokenizer(
							st.nextToken(), ":");
					StringTokenizer index = new StringTokenizer(
							feature.nextToken(), ",");
					int indexlength = index.countTokens();
					for (int j = 0; j < indexlength; j++) {
						tempt.add((int) atof(index.nextToken()));
					}
					datahashmap.put(tempt, atof(feature.nextToken()));
				}
				pdthashdata.add(datahashmap);
			}
			fp.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	private Vector<svm_node[]> read_data() throws IOException
	{
		BufferedReader fp;
		fp = new BufferedReader(new FileReader(
				"..\\SampleTestTensorPrint.txt"));Vector<Double> vy = new Vector<Double>();
		Vector<svm_node[]> X = new Vector<svm_node[]>();
		int max_index = 0;

		while(true)
		{
			String line = fp.readLine();
			if(line == null) break;

			StringTokenizer st = new StringTokenizer(line," \t\n\r\f:");

			vy.addElement(atof(st.nextToken()));
			int m = st.countTokens()/2;
			svm_node[] x = new svm_node[m];
			for(int j=0;j<m;j++)
			{
				x[j] = new svm_node();
				x[j].index = atoi(st.nextToken());
				x[j].value = atof(st.nextToken());
			}
			if(m>0) max_index = Math.max(max_index, x[m-1].index);
			X.addElement(x);
		}
		
		prob = new svm_problem();
		prob.l = vy.size();
		prob.x = new svm_node[prob.l][];
		for(int i=0;i<prob.l;i++)
			prob.x[i] = X.elementAt(i);
		prob.y = new double[prob.l];
		for(int i=0;i<prob.l;i++)
			prob.y[i] = vy.elementAt(i);

		if(param.gamma == 0 && max_index > 0)
			param.gamma = 1.0/max_index;

		if(param.kernel_type == svm_parameter.PRECOMPUTED)
			for(int i=0;i<prob.l;i++)
			{
				if (prob.x[i][0].index != 0)
				{
					System.err.print("Wrong kernel matrix: first column must be 0:sample_serial_number\n");
					System.exit(1);
				}
				if ((int)prob.x[i][0].value <= 0 || (int)prob.x[i][0].value > max_index)
				{
					System.err.print("Wrong input format: sample_serial_number out of range\n");
					System.exit(1);
				}
			}

		fp.close();
		
		return X;
		
	}
	
	private static int atoi(String s)
	{
		return Integer.parseInt(s);
	}
	
	private void read_predictdata(){
		svm_node[] x ;
		try{
			BufferedReader fp;
			fp = new BufferedReader(new FileReader("..\\SampleTestTensorPrint2.txt"));
			String line;
			line = fp.readLine();
			
		}catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	
	private double read_vector(int x,int y,int z) {
		Vector<Double> vx = new Vector<Double>();
		Vector<Double> vy = new Vector<Double>();
		Vector<Double> vz = new Vector<Double>(); 
		int flag = 1;
		try {
			
			BufferedReader fp;
			fp = new BufferedReader(new FileReader("..\\predict.txt"));

			//����һ�е�flag
			String line;
			line = fp.readLine();
			StringTokenizer st = new StringTokenizer(line, " \t\n\r\f:");
			st.nextToken();
			flag = Integer.parseInt(st.nextToken());
			System.out.println("flag:"+ flag);

			//���ڶ��е�hashvx����
			String line1;
			line1 = fp.readLine();
			StringTokenizer st1 = new StringTokenizer(line1, " \t\n\r\f:");
			int m1 = st1.countTokens() / 2;
			for (int j = 0; j < m1; j++) {
				st1.nextToken();
				double s = atof(st1.nextToken());
				vx.add(s);
			}

			//�������е�hashvy����
			String line2;
			line2 = fp.readLine();
			StringTokenizer st2 = new StringTokenizer(line2, " \t\n\r\f:");
			int m2 = st2.countTokens() / 2;
			for (int j = 0; j < m2; j++) {
				st2.nextToken();
				double s = atof(st2.nextToken());
				vy.add(s);
			}
			
			//�������е�hashvz����
			String line3;
			line3 = fp.readLine();
			StringTokenizer st3 = new StringTokenizer(line3, " \t\n\r\f:");
			int m3 = st3.countTokens() / 2;
			for (int j = 0; j < m3; j++) {
				st3.nextToken();
				double s = atof(st3.nextToken());
				vz.add(s);
			}
			
			//�������е�betaֵ
			String line4;
			line4 = fp.readLine();
			beta = atof(line4);
			fp.close();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		double bata = datapretreatment(vx,vy,vz,flag);
		
		return bata;
	}
	
	private double datapretreatment(Vector<Double> vx,Vector<Double> vy,Vector<Double> vz,int flag){
		//��Ԥ��������һ��Ԥ������������ά�����������ҷ���bata����ʽ�Ƶ��е�||v||*||v||��ֵ
		Vector<Double> hashvx = new Vector<Double>();
		Vector<Double> hashvy = new Vector<Double>();
		Vector<Double> hashvz = new Vector<Double>();
		hashvx = vx;
		hashvy = vy;
		hashvz = vz;
		double bata1 = 0 ;
			//��������������flag = 1 �Ǳ���x�ף���y��z�׾��н�ά����
			if (flag == 1) {
				try {
					File f = new File("..\\SampleTestTensorPrint2.txt");
					if (f.exists())
						f.delete();
					FileWriter output = new FileWriter("..\\SampleTestTensorPrint2.txt");
					BufferedWriter bw = new BufferedWriter(output);
					for (int i = 0; i < pdthashflag.size(); i++) {
						bw.write(Double.toString(pdthashflag.get(i)));
						HashMap<Vector<Integer>, Double> resulthashmap = new HashMap<Vector<Integer>, Double>();
						resulthashmap = HashMultipl(HashMultipl(pdthashdata.get(i), 3, hashvz),2,hashvy);
						if (i == 0) {
							for(int sample = 1; sample <= hashvx.size(); sample++){
								int written = 0;
								Iterator iter = resulthashmap.entrySet().iterator();
								while (iter.hasNext()) {
									Map.Entry<Vector<Integer>, Double> entry = (Map.Entry) iter.next();
									Vector<Integer> key = (Vector) entry.getKey();
									if(sample == key.get(0)){
										bw.write(" " + Integer.toString(sample) + ":" + Double.toString(entry.getValue()));
										written = 1;
									}
								}
								if(written == 0){
									bw.write(" " + Integer.toString(sample) + ":" + Integer.toString(0));
								}								
							}
							bw.newLine();
						}
						if (i != 0) {
							//Ϊ��ʹ��ÿ�������������ܰ���˳���������д��
							for (int sample = 1; sample <= hashvx.size(); sample++) {
								Iterator iter = resulthashmap.entrySet().iterator();
								while (iter.hasNext()) {
									Map.Entry<Vector<Integer>, Double> entry = (Map.Entry) iter.next();
									Vector<Integer> key = (Vector) entry.getKey();
									if(sample == key.get(0)){
										bw.write(" " + Integer.toString(sample) + ":" + Double.toString(entry.getValue()));
									}
								}
							}
							bw.newLine();
						}
					}
					bw.close();
					double sum1 = 0;
					for(int j = 0 ; j < vx.size();j++){
						sum1 = vz.get(j)*vz.get(j) + sum1;
					}
					double sum2 = 0;
					for(int j = 0 ; j < vx.size();j++){
						sum2 = vy.get(j)*vy.get(j) + sum2;
					}
					bata1 = sum1*sum2;
				} catch (IOException e) {
					e.printStackTrace();
				}
				
			}
			
			
			//��������������flag = 2 �Ǳ���y�ף���x��z�׾��н�ά����
			if (flag == 2) {
				try {
					File f = new File("..\\SampleTestTensorPrint2.txt");
					if (f.exists())
						f.delete();
					FileWriter output = new FileWriter("..\\SampleTestTensorPrint2.txt");
					BufferedWriter bw = new BufferedWriter(output);
					for (int i = 0; i < pdthashflag.size(); i++) {
						bw.write(Double.toString(pdthashflag.get(i)));
						HashMap<Vector<Integer>, Double> resulthashmap = new HashMap<Vector<Integer>, Double>();
						resulthashmap = HashMultipl(HashMultipl(pdthashdata.get(i), 3, hashvz),1,hashvx);
						if (i == 0) {
							for(int sample = 1; sample <= hashvy.size(); sample++){
								int written = 0;
								Iterator iter = resulthashmap.entrySet().iterator();
								while (iter.hasNext()) {
									Map.Entry<Vector<Integer>, Double> entry = (Map.Entry) iter.next();
									Vector<Integer> key = (Vector) entry.getKey();
									if(sample == key.get(0)){
										bw.write(" " + Integer.toString(sample) + ":" + Double.toString(entry.getValue()));
										written = 1;
									}
								}
								if(written == 0){
									bw.write(" " + Integer.toString(sample) + ":" + Integer.toString(0));
								}								
							}
							bw.newLine();
						}
						if (i != 0) {
							for (int sample = 1; sample <= hashvy.size(); sample++) {
								Iterator iter = resulthashmap.entrySet().iterator();
								while (iter.hasNext()) {
									Map.Entry<Vector<Integer>, Double> entry = (Map.Entry) iter.next();
									Vector<Integer> key = (Vector) entry.getKey();
									if(sample == key.get(0)){
										bw.write(" " + Integer.toString(sample) + ":" + Double.toString(entry.getValue()));
									}
								}
							}
							bw.newLine();
						}
					}
					bw.close();
					double sum1 = 0;
					for(int j = 0 ; j < vz.size();j++){
						sum1 = vz.get(j)*vz.get(j) + sum1;
					}
					double sum2 = 0;
					for(int j = 0 ; j < vx.size();j++){
						sum2 = vx.get(j)*vx.get(j) + sum2;
					}
					bata1 = sum1*sum2;
				} catch (IOException e) {
					e.printStackTrace();
				}
			
			}
			
			//��������������flag = 3 �Ǳ���z�ף���x��y�׾��н�ά����
			if (flag == 3) {
				try {
					File f = new File("..\\SampleTestTensorPrint2.txt");
					if (f.exists())
						f.delete();
					FileWriter output = new FileWriter("..\\SampleTestTensorPrint2.txt");
					BufferedWriter bw = new BufferedWriter(output);
					for (int i = 0; i < pdthashflag.size(); i++) {
						bw.write(Double.toString(pdthashflag.get(i)));
						HashMap<Vector<Integer>, Double> resulthashmap = new HashMap<Vector<Integer>, Double>();
						resulthashmap = HashMultipl(HashMultipl(pdthashdata.get(i), 2, hashvy),1,hashvx);
						if (i == 0) {
							for(int sample = 1; sample <= hashvz.size(); sample++){
								int written = 0;
								Iterator iter = resulthashmap.entrySet().iterator();
								while (iter.hasNext()) {
									Map.Entry<Vector<Integer>, Double> entry = (Map.Entry) iter.next();
									Vector<Integer> key = (Vector) entry.getKey();
									if(sample == key.get(0)){
										bw.write(" " + Integer.toString(sample) + ":" + Double.toString(entry.getValue()));
										written = 1;
									}
								}
								if(written == 0){
									bw.write(" " + Integer.toString(sample) + ":" + Integer.toString(0));
								}								
							}
							bw.newLine();
							
							
							
						}
						if (i != 0) {
							for (int sample = 1; sample <= hashvz.size(); sample++) {
								Iterator iter = resulthashmap.entrySet().iterator();
								while (iter.hasNext()) {
									Map.Entry<Vector<Integer>, Double> entry = (Map.Entry) iter.next();
									Vector<Integer> key = (Vector) entry.getKey();
									if(sample == key.get(0)){
										bw.write(" " + Integer.toString(sample) + ":" + Double.toString(entry.getValue()));
									}
								}
							}
							bw.newLine();
						}
						
					}
					bw.close();
					
					double sum1 = 0;
					for(int j = 0 ; j < vx.size();j++){
						sum1 = vx.get(j)*vx.get(j) + sum1;
					}
					double sum2 = 0;
					for(int j = 0 ; j < vy.size();j++){
						sum2 = vy.get(j)*vy.get(j) + sum2;
					}
					bata1 = sum1*sum2;
				} catch (IOException e) {
					e.printStackTrace();
				}
				
				
			}
			return bata1;
		
		
	}
	
	
	private HashMap<Vector<Integer>, Double> HashMultipl(HashMap<Vector<Integer>, Double> hashmap, int l, Vector<Double> a) {
		HashMap<Vector<Integer>, Double> Result = new HashMap<Vector<Integer>, Double>();
		Iterator iter = hashmap.entrySet().iterator();
		while (iter.hasNext()) {
			Double tempval;
			Vector<Integer> tempKey = new Vector<Integer>();
			Map.Entry entry = (Map.Entry) iter.next();
			Vector key = (Vector) entry.getKey();
			Double val = (Double) entry.getValue();
			// ���潵ά���valueֵ
			tempval = val * a.get((Integer) key.get(l - 1) - 1);
			// ���潵ά���keyֵ
			for (int i = 0; i < l - 1; i++) {
				tempKey.add((Integer) key.get(i));
			}
			for (int i = l; i < key.size(); i++) {
				tempKey.add((Integer) key.get(i));
			}
			// ���µõ���valueֵ��keyֵ����HashMap��
			if (Result.containsKey(tempKey)) {
				Double Val0 = Result.get(tempKey);
				Result.put(tempKey, tempval + Val0);
			} else {
				Result.put(tempKey, tempval);
			}
		}
		return Result;
	}
	
	
	
	private void printvector(double[] vx2){
		int len = vx2.length;
		for(int i =0;i<(len-1);i++){
			System.out.print(i+1);
			System.out.print(":"+vx2[i]);
		}
		System.out.println(len+":"+vx2[len-1]);
	}
	
	
	private void predict(double bata) {
		double sum = 0;
		
		model = svm_train.get_svm_model();
		
		System.out.println("bata��ֵ��"+ bata);
		
		//��bata��bֵд��bata.txt�з���libsvm�ж������
		try {
			File f = new File("..\\bata.txt");
			if (f.exists())
				f.delete();
			FileWriter output = new FileWriter("..\\bata.txt");
			BufferedWriter bw = new BufferedWriter(output);
			        bw.write(Double.toString(bata));
					bw.newLine();
					bw.write(Double.toString(b));
					bw.newLine();
					bw.close();
					
				}catch (IOException e) {
					e.printStackTrace();
				}
		
		
		

		svm_predict p = new svm_predict();
		String[] parg = { "..\\SampleTestTensorPrint2.txt", // ����Ǵ�Ų�������
				"..\\recsys_model.txt", // ���õ���ѵ���Ժ��ģ��
				"..\\out_r.txt" }; // ���ɵĽ�����ļ���·��
		
		try {
			p.main(parg);
		} catch (IOException e) {
			// TODO �Զ����ɵ� catch ��
			e.printStackTrace();
		}
		

		
		
		
		
		
//		for (int i = 0; i < pdthashflag.size(); i++) {
//			double predict_y = 0;
//			HashMap<Vector<Integer>, Double> pdtmap = new HashMap<Vector<Integer>, Double>();
//			pdtmap = pdthashdata.get(i);
//			Iterator iter = pdtmap.entrySet().iterator();
//			while (iter.hasNext()) {
//				Map.Entry<Vector<Integer>, Double> entry = (Map.Entry) iter.next();		
//				predict_y = predict_y + entry.getValue()*vx[(entry.getKey()).get(0)-1]*vy[(entry.getKey()).get(1)-1]*vz[(entry.getKey()).get(2)-1];
//			}			
//			System.out.print("��ʵֵ��"+pdthashflag.get(i)+"  ");
//			System.out.println("Ԥ��ֵ��"+(predict_y+b));			
//			sum = sum + Math.abs((predict_y+b) - pdthashflag.get(i));	
//		}
//		System.out.println("MAE:"+sum/pdthashflag.size());
//		
//		
	}
	
	//Stringת��ΪDouble��
	private double atof(String s) {
		double d = Double.valueOf(s).doubleValue();
		if (Double.isNaN(d) || Double.isInfinite(d)) {
			System.err.print("NaN or Infinity in input\n");
			System.exit(1);
		}
		return (d);
	}
	
}
