package us.bosch.rtc;

import java.io.IOException;

import edu.stanford.nlp.classify.ColumnDataClassifier;
import edu.stanford.nlp.tagger.maxent.MaxentTagger;

public class StanfordClassifier {

	/**
	 * @param args
	 * @throws ClassNotFoundException 
	 * @throws IOException 
	 */
	public static void main(String[] args) throws IOException, ClassNotFoundException {
		// TODO Auto-generated method stub
		// Initialize the tagger
		String options = "-trainFile 20news-bydate-devtrain-stanford-classifier.txt -testFile 20news-bydate-devtest-stanford-classifier.txt -2.useSplitWords -2.splitWordsRegexp \"\\s+\"";
		ColumnDataClassifier classifer = new ColumnDataClassifier(options);
		
		
		}
}
