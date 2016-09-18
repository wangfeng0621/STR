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
import service.svm_train;


public class MatrixMain {
	private Vector<Double> hashflag = new Vector<Double>();
	//Vector�У���������˳��Ϊ��x�׵��±꣬y�׵��±꣬z�׵��±�
	private Vector<HashMap<Vector<Integer>, Double>> hashdata = new Vector<HashMap<Vector<Integer>, Double>>();
	
	public static void main(String[] args){
		MatrixMain matr = new MatrixMain();
		matr.readHashData();
		matr.train(3,4,5);
	}
	
	public void trainMethod(int x,int y,int z){
		readHashData();
		train(x, y, z);
	}
	
	private void train(int x, int y ,int z){  
		double beta;  
		//vx,vy,vz�ֱ�������е�x��y��z���Ӧ�׽��н�ά
		Vector<Double> newhashvx = new Vector<Double>();   
		Vector<Double> newhashvy = new Vector<Double>();
		Vector<Double> newhashvz = new Vector<Double>();
		Vector<Double> hashvx = new Vector<Double>();
		Vector<Double> hashvy = new Vector<Double>();
		Vector<Double> hashvz = new Vector<Double>();
		//x���ᣩ�׵�ά������
		for(int i = 0;i < x;i++){
			hashvx.add(1.0);
		}
		//y���ᣩ�׵�ά������
		for(int i = 0;i < y;i++){
			hashvy.add(1.0);
		}
		//z���ᣩ�׵�ά������
		for(int i = 0;i < z;i++){
			hashvz.add(1.0);
		}
		double yuzhi = 0.1;
		int flag = 1;                              
		int iteration = 1; 
		String kern = "0";
		String []argss={"-s","3","-t",kern,"-g","0.1","-d","3","-r","0","-n","0.5","-m","1",
				"-c","1","-p","0.0125","-h","0","-b","0","..\\recsys_train2.txt",   
				"..\\recsys_model.txt"};
		while (true) {
			//��������������flag = 1 �Ǳ���x�ף���y��z�׾��н�ά����
			if (flag == 1) {
				
				System.out.println("**********************************");
				System.out.println("��������iteration:"+iteration+"���ڱ�ǩflag��"+flag);
				try {
					File f = new File("..\\recsys_train2.txt");
					if (f.exists())
						f.delete();
					FileWriter output = new FileWriter("..\\recsys_train2.txt");
					BufferedWriter bw = new BufferedWriter(output);
					for (int i = 0; i < hashflag.size(); i++) {
						bw.write(Double.toString(hashflag.get(i)));
						HashMap<Vector<Integer>, Double> resulthashmap = new HashMap<Vector<Integer>, Double>();
						resulthashmap = HashMultipl(HashMultipl(hashdata.get(i), 3, hashvz),2,hashvy);
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
				} catch (IOException e) {
					e.printStackTrace();
				}
				beta = norm(hashvy)*norm(hashvz);
				System.out.println("�����ڲ����");
				try{
					svm_train.main(argss, beta);
				}catch(IOException e){
					e.printStackTrace();
				}
				System.out.println("�����ڲ����");
				newhashvx = read_vector();
				iteration = iteration + 1;
				System.out.println("����ǰ���Ż�������"+norm(hashvx) * norm(hashvy) * norm(hashvz));
				System.out.println("��������Ż�������"+norm(newhashvx) * norm(hashvy) * norm(hashvz));	
				//�ж��Ƿ�����������
				//if (iteration > 0) {
				//if (((norm(newhashvy) * norm(hashvx) - norm(hashvx) * norm(hashvy)) >= 0) && (iteration > 0)) {
				if (((norm(newhashvx) * norm(newhashvy) * norm(newhashvz)- norm(hashvx) * norm(hashvy) * norm(hashvz)) >= 0) && (Math.abs(norm(newhashvx) * norm(newhashvy) * norm(newhashvz)- norm(hashvx) * norm(hashvy) * norm(hashvz)) < yuzhi)) {
				//if ((Math.abs(norm(newhashvx) * norm(newhashvy) * norm(newhashvz)- norm(hashvx) * norm(hashvy) * norm(hashvz)) < yuzhi)) {

				    System.out.println("����������");
					System.out.println("����ǰ���Ż�������"+norm(hashvx) * norm(hashvy) * norm(hashvz));
					System.out.println("��������Ż�������"+norm(newhashvx) * norm(hashvy) * norm(hashvz));
					hashvx = (Vector<Double>) newhashvx.clone();
					System.out.println("дpredict.txt�ļ�");
					try {
						File f = new File("..\\predict.txt");
						if (f.exists())
							f.delete();
						FileWriter output = new FileWriter("..\\predict.txt");
						BufferedWriter bw = new BufferedWriter(output);
						bw.write("flag:"+flag);
						bw.newLine();
						for (int i = 0; i < hashvx.size(); i++) {
							bw.write(i + ":"+ hashvx.get(i)+" ");
						}
						bw.newLine();
						for (int i = 0; i < hashvy.size(); i++) {
							bw.write(i + ":"+ hashvy.get(i)+" ");
						}
						bw.newLine();
						for (int i = 0; i < hashvz.size(); i++){
							bw.write(i + ":"+ hashvz.get(i)+" ");
						}
						bw.newLine();
						bw.write(beta+"");
						bw.close();
					} catch (IOException e) {
						e.printStackTrace();
					}
					break;
				} else {
					hashvx = (Vector<Double>) newhashvx.clone();
				}
				flag = 2;
			}
			
			
			//��������������flag = 2 �Ǳ���y�ף���x��z�׾��н�ά����
			if (flag == 2) {
				
				System.out.println("**********************************");
				System.out.println("��������iteration:"+iteration+"���ڱ�ǩflag��"+flag);
				try {
					File f = new File("..\\recsys_train2.txt");
					if (f.exists())
						f.delete();
					FileWriter output = new FileWriter("..\\recsys_train2.txt");
					BufferedWriter bw = new BufferedWriter(output);					
					for (int i = 0; i < hashflag.size(); i++) {
						bw.write(Double.toString(hashflag.get(i)));
						HashMap<Vector<Integer>, Double> resulthashmap = new HashMap<Vector<Integer>, Double>();
						resulthashmap = HashMultipl(HashMultipl(hashdata.get(i), 3, hashvz),1,hashvx);
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
				} catch (IOException e) {
					e.printStackTrace();
				}
				beta = norm(hashvx)*norm(hashvz);
				System.out.println("�����ڲ����");
				try{
					svm_train.main(argss, beta);
				}catch(IOException e){
					e.printStackTrace();
				}
				System.out.println("�����ڲ����");
				newhashvy = read_vector();
				iteration =iteration +1;
				System.out.println("����ǰ���Ż�������"+norm(hashvx) * norm(hashvy) * norm(hashvz));
				System.out.println("��������Ż�������"+norm(hashvx) * norm(newhashvy) * norm(hashvz));
				//�ж��Ƿ�����������
				//if (((norm(newhashvx) * norm(hashvy) - norm(hashvx) * norm(hashvy) ) >= 0) && (iteration > 2)) {
				if (((norm(newhashvx) * norm(newhashvy) * norm(newhashvz)- norm(hashvx) * norm(hashvy) * norm(hashvz)) >= 0) && (Math.abs(norm(newhashvx) * norm(newhashvy) * norm(newhashvz)- norm(hashvx) * norm(hashvy) * norm(hashvz)) < yuzhi)) {
			    //if ((Math.abs(norm(newhashvx) * norm(newhashvy) * norm(newhashvz)- norm(hashvx) * norm(hashvy) * norm(hashvz)) < yuzhi)) {

					System.out.println("����������");
					System.out.println("����ǰ���Ż�������"+norm(hashvx) * norm(hashvy) * norm(hashvz));
					System.out.println("��������Ż�������"+norm(hashvx) * norm(newhashvy) * norm(hashvz));
					hashvy = (Vector<Double>) newhashvy.clone();
					try {
						File f = new File("..\\predict.txt");
						if (f.exists())
							f.delete();
						FileWriter output = new FileWriter(
								"..\\predict.txt");
						BufferedWriter bw = new BufferedWriter(output);
						bw.write("flag:"+flag);
						bw.newLine();
						for (int i = 0; i < hashvx.size(); i++) {
							bw.write(i + ":"+ hashvx.get(i)+" ");
						}
						bw.newLine();
						for (int i = 0; i < hashvy.size(); i++) {
							bw.write(i + ":"+ hashvy.get(i)+" ");
						}
						bw.newLine();
						for (int i = 0; i < hashvz.size(); i++) {
							bw.write(i + ":"+ hashvz.get(i)+" ");
						}
						bw.newLine();
						bw.write(beta+"");
						bw.close();
					} catch (IOException e) {
						e.printStackTrace();
					}
					break;
				} else {
					hashvy = (Vector<Double>) newhashvy.clone();
				}
				flag = 3;
			}
			
			//��������������flag = 3 �Ǳ���z�ף���x��y�׾��н�ά����
			if (flag == 3) {
				
				System.out.println("**********************************");
				System.out.println("��������iteration:"+iteration+"���ڱ�ǩflag��"+flag);
				try {
					File f = new File("..\\recsys_train2.txt");
					if (f.exists())
						f.delete();
					FileWriter output = new FileWriter("..\\recsys_train2.txt");
					BufferedWriter bw = new BufferedWriter(output);
					for (int i = 0; i < hashflag.size(); i++) {
						bw.write(Double.toString(hashflag.get(i)));
						HashMap<Vector<Integer>, Double> resulthashmap = new HashMap<Vector<Integer>, Double>();
						resulthashmap = HashMultipl(HashMultipl(hashdata.get(i), 2, hashvy),1,hashvx);
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
				} catch (IOException e) {
					e.printStackTrace();
				}
				beta = norm(hashvx)*norm(hashvy);
				System.out.println("�����ڲ����");
				try{
					svm_train.main(argss, beta);
				}catch(IOException e){
					e.printStackTrace();
				}
				System.out.println("�����ڲ����");
				newhashvz = read_vector();
				iteration =iteration +1;
				System.out.println("����ǰ���Ż�������"+norm(hashvx) * norm(hashvy) * norm(hashvz));
				System.out.println("��������Ż�������"+norm(hashvx) * norm(hashvy) * norm(newhashvz));
				//�ж��Ƿ�����������
				//if (((norm(newhashvx) * norm(hashvy) - norm(hashvx) * norm(hashvy) ) >= 0) && (iteration > 2)) {
				if (((norm(newhashvx) * norm(newhashvy) * norm(newhashvz)- norm(hashvx) * norm(hashvy) * norm(hashvz)) >= 0) && (Math.abs(norm(newhashvx) * norm(newhashvy) * norm(newhashvz)- norm(hashvx) * norm(hashvy) * norm(hashvz)) < yuzhi)) {
				//if ((Math.abs(norm(newhashvx) * norm(newhashvy) * norm(newhashvz)- norm(hashvx) * norm(hashvy) * norm(hashvz)) < yuzhi)) {

				    System.out.println("����������");
					System.out.println("����ǰ���Ż�������"+norm(hashvx) * norm(hashvy) * norm(hashvz));
					System.out.println("��������Ż�������"+norm(hashvx) * norm(hashvy) * norm(newhashvz));
					hashvz = (Vector<Double>) newhashvz.clone();
					try {
						File f = new File("..\\predict.txt");
						if (f.exists())
							f.delete();
						FileWriter output = new FileWriter(
								"..\\predict.txt");
						BufferedWriter bw = new BufferedWriter(output);
						bw.write("flag:"+flag);
						bw.newLine();
						for (int i = 0; i < hashvx.size(); i++) {
							bw.write(i + ":"+ hashvx.get(i)+" ");
						}
						bw.newLine();
						for (int i = 0; i < hashvy.size(); i++) {
							bw.write(i + ":"+ hashvy.get(i)+" ");
						}
						bw.newLine();
						for (int i = 0; i < hashvz.size(); i++) {
							bw.write(i + ":"+ hashvz.get(i)+" ");
						}
						bw.newLine();
						bw.write(beta+"");
						bw.close();
					} catch (IOException e) {
						e.printStackTrace();
					}
					break;
				} else {
					hashvz = (Vector<Double>) newhashvz.clone();
				}
				flag = 1;
			}
		}
		
	}
		
	// ��ȡ����svm����������
	private Vector<Double> read_vector() {
		Vector<Double> hashvector = new Vector<Double>();
		try {
			BufferedReader fp;
			fp = new BufferedReader(new FileReader("..\\vector.txt"));// �������ѵ�����ݣ��������ݣ��ĵ�ַ
			while (true) { // ���ݾ�������ĳ���ѭ��
				String line;
				line = fp.readLine();
				if (line == null)
					break;
				StringTokenizer st = new StringTokenizer(line, " \t\n\r\f");
				int length = st.countTokens();
				for (int i = 0; i < length; i++) {
					StringTokenizer feature = new StringTokenizer(
							st.nextToken(), ":");
					feature.nextToken();
					hashvector.add(atof(feature.nextToken()));
				}
			}
			fp.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
		return hashvector;
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

	private void readHashData() {
		try {
			BufferedReader fp;
			fp = new BufferedReader(new FileReader("D:\\STR\\SampleTrainTensorPrint.txt"));
			while (true) { 
				HashMap<Vector<Integer>, Double> datahashmap = new HashMap<Vector<Integer>, Double>();
				String line;
				line = fp.readLine();
				if (line == null)
					break;
				StringTokenizer st = new StringTokenizer(line, " \t\n\r\f");
				int length = st.countTokens();
				//hashflag��һ��vector���ͣ���������ÿһ��������Rating
				hashflag.add(atof(st.nextToken()));
				for (int i = 1; i < length; i++) {
					Vector<Integer> tempt = new Vector<Integer>();
					//feature�������������ԣ��ã��ָ��index��value
					StringTokenizer feature = new StringTokenizer(st.nextToken(), ":");
					//��index�����ã��ָ�õ�index
					StringTokenizer index = new StringTokenizer(feature.nextToken(), ",");
					int indexlength = index.countTokens();
					for (int j = 0; j < indexlength; j++) {
						tempt.add((int) atof(index.nextToken()));
					}
					datahashmap.put(tempt, atof(feature.nextToken()));
				}
				hashdata.add(datahashmap);
			}
			fp.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	
	//int l��ʾ�Ƕ�l�׽��н�ά
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
	
	private double norm(Vector <Double> vec) {
		double norm = 0.0;
		for (int i = 0; i < vec.size(); i++) {
			norm = norm + vec.get(i)  * vec.get(i);
		}		
		return norm;
	}
	
	static void printvector(double[] vector){
		int len = vector.length;
		for(int i =0;i<(len-1);i++){
			System.out.print(i+1);
			System.out.print(":"+vector[i]);
		}
		System.out.println(len+":"+vector[len-1]);
	}


}
