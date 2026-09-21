export interface ExampleArticle {
  id: string
  label: string
  title: string
  text: string
}

/**
 * Ported from the project's existing Streamlit demo selector (app.py),
 * plus one additional real-leaning example for balance.
 */
export const EXAMPLE_ARTICLES: ExampleArticle[] = [
  {
    id: 'example-1',
    label: 'Example 1',
    title: 'Government Hides Shocking Truth About UFOs',
    text: 'SHOCKING: Government secretly admits to hiding UFO technology! Deep state operatives have confirmed that the mainstream media has been suppressing this information for decades. Anonymous sources reveal that miracle cures are being kept from the public to protect Big Pharma profits.',
  },
  {
    id: 'example-2',
    label: 'Example 2',
    title: 'Fed holds rates steady, signals cautious approach',
    text: 'The Federal Reserve held interest rates steady on Wednesday and signaled it remains in no hurry to resume cutting borrowing costs, as policymakers evaluate the evolving economic outlook and the potential effects of new government policies. The central bank kept its benchmark overnight interest rate in the 4.25%-4.50% range.',
  },
  {
    id: 'example-3',
    label: 'Example 3',
    title: 'City council approves new transit funding after budget review',
    text: 'The city council voted 6-2 on Tuesday to approve additional funding for the regional transit authority, following a three-month budget review process. Officials said the funds will go toward maintenance of the existing bus fleet and planning for two new routes, with construction expected to begin next fiscal year pending state approval.',
  },
]

/**
 * Metrics reported from the model's training run. The deployed API does not
 * expose dataset size, ROC-AUC, or embedding dimension via /model/info, so
 * these are fixed figures from the project's training report rather than
 * live values. Do not present these as if they come from a live endpoint.
 */
export const MODEL_FACTS = {
  embeddingModel: 'Sentence Transformers — all-MiniLM-L6-v2',
  embeddingDimension: 384,
  datasetSize: '44,856 articles',
  testF1: 0.934,
  testRocAuc: 0.984,
  fallbackClassifier: 'Logistic Regression',
  fallbackVersion: 'v1.0',
}
