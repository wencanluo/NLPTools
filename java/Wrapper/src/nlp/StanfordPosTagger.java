package nlp;

import java.io.IOException;

import edu.stanford.nlp.tagger.maxent.MaxentTagger;

public class StanfordPosTagger {

	/**
	 * @param args
	 * @throws ClassNotFoundException 
	 * @throws IOException 
	 */
	public static void main(String[] args) throws IOException, ClassNotFoundException {
		// TODO Auto-generated method stub
		// Initialize the tagger
		String modelPath = "../../software/stanford-postagger-2013-04-04/models/english-bidirectional-distsim.tagger";
		MaxentTagger tagger = new MaxentTagger(modelPath);
		
		// The sample string
		String sample = "This is a sample text";
		// The tagged string
		String tagged = tagger.tagString(sample);
		// Output the result
		System.out.println(tagged);
	}

}
