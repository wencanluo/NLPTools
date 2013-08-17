package com.bosch.emotiondetection.weka;

import java.util.ArrayList;
import java.util.Hashtable;
import java.util.Iterator;

import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;



public class WekaWriter {
	public ArrayList<Attribute> buildAttrs(ArrayList<String> attrNames,ArrayList<Object> attrTypes) {
		ArrayList<Attribute> attrs = new ArrayList<Attribute>();
		for(int i = 0;i<attrNames.size();i++) {
			String attrName = attrNames.get(i);
			Object attrType = attrTypes.get(i);
			if(attrType.getClass().equals(Integer.class)||attrType.getClass().equals(Double.class)) {
				//NUMERIC
				attrs.add(new Attribute(attrName));
			} else if(attrType.getClass().equals(ArrayList.class)) {
				//NOMINAL
				ArrayList<String> types = (ArrayList<String>)attrType;
				attrs.add(new Attribute(attrName,types));
			}
		}
		return attrs;
	}
	
	public Instances buildInstances(ArrayList<Attribute> attrs,ArrayList<Hashtable<String,Object>> data) {
		Instances instances = new Instances("data",attrs,0);
		Hashtable<String,Object> firstAttr = data.get(0);
		Iterator<String> it = firstAttr.keySet().iterator();
		while(it.hasNext()) {
			String id = it.next();
			double[] vals = new double[data.size()];
			for(int i = 0;i<data.size();i++) {
				Object val = data.get(i).get(id);
				if(val.getClass().equals(Integer.class)) {
					vals[i] = 1.0*(Integer)val;
				} else if(val.getClass().equals(Double.class)) {
					vals[i] = (Double)val;
				} else {
					vals[i] = attrs.get(i).indexOfValue((String)val);
				}
			}
			instances.add(new DenseInstance(1.0,vals));
		}
		System.out.println(instances);
		return instances;
	}
}
