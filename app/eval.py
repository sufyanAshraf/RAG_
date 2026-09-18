from typing import List, Callable, Any

from pydantic import BaseModel, Field
from langsmith import Client
from langsmith.evaluation import evaluate
from .evalDataPrepration import retrieve

class RAGEvaluator:
    """
    LangSmith-based evaluator for a RAG pipeline.

    Evaluates:
    - Context Precision
    - Faithfulness
    - Answer Relevancy
    - Context Recall
    """

    class Score(BaseModel):
        score: float = Field(
            description="Score between 0.0 and 1.0"
        )
        reasoning: str = Field(
            description="Brief justification for the score"
        )

    def __init__(self, eval_queries: List[dict], model : Any, key: str, db: Any, model_class: Any, processor: Any,  dataset_name: str = "RAG_Eval_Dataset1"):
        """
        Args:
            eval_queries: List of dictionaries containing:
                {
                    "query": "...",
                    "reference": "..."
                }
            model: LLM used as the evaluator/judge.
            key: Langsmith api key
            dataset_name: Name of the LangSmith evaluation dataset.
        """

        self.eval_queries = eval_queries 
        self.client = Client(api_key=key)
        self.dataset_name = dataset_name
        self.db = db
        self.model_class = model_class
        self.processor = processor
        # LLM used for judging
        self.judge_llm = model
        self.structured_judge = self.judge_llm.with_structured_output(
            self.Score
        )

        # Create or load dataset
        self.dataset = self._setup_dataset()

        # Evaluators
        self.evaluators = [
            self.context_precision_evaluator,
            self.faithfulness_evaluator,
            self.answer_relevancy_evaluator,
            self.context_recall_evaluator,
        ]

    # -----------------------------------------------------------
    # Dataset
    # -----------------------------------------------------------

    def _setup_dataset(self):
        """Create or load the LangSmith evaluation dataset."""

        if self.client.has_dataset( dataset_name=self.dataset_name ):
            dataset = self.client.read_dataset(dataset_name=self.dataset_name )
        else:
            dataset = self.client.create_dataset( dataset_name=self.dataset_name)

        self.client.create_examples(
            inputs=[
                {"user_input": item["query"]}
                for item in self.eval_queries
            ],
            outputs=[
                {"reference": item["reference"]}
                for item in self.eval_queries
            ],
            dataset_id=dataset.id,
        )

        return dataset


         
    # -----------------------------------------------------------
    # RAG Target
    # -----------------------------------------------------------
    
    def rag_target(self, inputs: dict) -> dict:
        """
        Calls the live RAG pipeline.

        LangSmith passes:
            {"user_input": "..."}

        Returns:
            response
            retrieved_contexts
        """

        query = inputs["user_input"]

        response, contexts = retrieve(query, self.db, self.model_class, self.processor)
          
        return {
            "response": response,
            "retrieved_contexts": contexts,
        }

    # -----------------------------------------------------------
    # Helper
    # -----------------------------------------------------------

    @staticmethod
    def _contexts_block(contexts: List[str]) -> str:
        """Format retrieved contexts for the judge LLM."""

        return "\n".join(
            f"[{i + 1}] {context}"
            for i, context in enumerate(contexts)
        )

    # -----------------------------------------------------------
    # Faithfulness
    # -----------------------------------------------------------

    def faithfulness_evaluator(self, run, example) -> dict:
        """
        Measures whether the generated answer is supported
        by the retrieved context.
        """

        prompt = f"""
            You are grading whether a RESPONSE is faithful to the RETRIEVED CONTEXT.

            Faithfulness = the proportion of claims in the response that are
            directly supported by the context.

            Any claim not supported by the context is considered a hallucination
            and should lower the score.

            RETRIEVED CONTEXT:
            {self._contexts_block(run.outputs["retrieved_contexts"])}

            RESPONSE:
            {run.outputs["response"]}

            Score from 0.0 to 1.0.

            0.0 = response is unsupported/hallucinated
            1.0 = response is fully grounded in the retrieved context
            """

        result = self.structured_judge.invoke(prompt)

        return {
            "key": "faithfulness",
            "score": result.score,
            "comment": result.reasoning,
        }

    # -----------------------------------------------------------
    # Answer Relevancy
    # -----------------------------------------------------------

    def answer_relevancy_evaluator(self,run, example ) -> dict:
        """
        Measures whether the answer directly and completely
        addresses the user's question.
        """

        prompt = f"""
        You are grading ANSWER RELEVANCY.

        Judge how well the RESPONSE addresses the QUESTION.

        Penalize:
        - incomplete answers
        - off-topic information
        - unnecessary padding
        - failure to answer the actual question

        QUESTION:
        {example.inputs["user_input"]}

        RESPONSE:
        {run.outputs["response"]}

        Score from 0.0 to 1.0.

        0.0 = irrelevant/off-topic
        1.0 = directly and fully answers the question
        """

        result = self.structured_judge.invoke(prompt)

        return {
            "key": "answer_relevancy",
            "score": result.score,
            "comment": result.reasoning,
        }

    # -----------------------------------------------------------
    # Context Precision
    # -----------------------------------------------------------

    def context_precision_evaluator( self, run, example) -> dict:
        """
        Measures whether relevant retrieved chunks are ranked
        near the top of the retrieval results.
        """

        prompt = f"""
        You are grading CONTEXT PRECISION.

        Given the QUESTION and REFERENCE answer, judge whether the
        retrieved context chunks that are truly relevant to answering
        the question are ranked near the top of the list.

        Lower-numbered chunks appear earlier in the retrieval ranking.

        Also consider whether irrelevant chunks are present.

        QUESTION:
        {example.inputs["user_input"]}

        REFERENCE ANSWER:
        {example.outputs["reference"]}

        RETRIEVED CONTEXT:
        {self._contexts_block(run.outputs["retrieved_contexts"])}

        Score from 0.0 to 1.0.

        0.0 = relevant chunks are missing/ranked last or mostly irrelevant
        1.0 = relevant chunks are present and ranked highest
        """

        result = self.structured_judge.invoke(prompt)

        return {
            "key": "context_precision",
            "score": result.score,
            "comment": result.reasoning,
        }

    # -----------------------------------------------------------
    # Context Recall
    # -----------------------------------------------------------

    def context_recall_evaluator(self, run, example) -> dict:
        """
        Measures whether the retrieved context contains the
        information needed to produce the reference answer.
        """

        prompt = f"""
        You are grading CONTEXT RECALL.

        Judge what proportion of the information in the REFERENCE ANSWER
        can be found or supported by the RETRIEVED CONTEXT.

        If key facts from the reference answer are completely missing
        from the retrieved context, context recall should be lower.

        REFERENCE ANSWER:
        {example.outputs["reference"]}

        RETRIEVED CONTEXT:
        {self._contexts_block(run.outputs["retrieved_contexts"])}

        Score from 0.0 to 1.0.

        0.0 = none of the reference information is present
        1.0 = all required reference information is present
        """

        result = self.structured_judge.invoke(prompt)

        return {
            "key": "context_recall",
            "score": result.score,
            "comment": result.reasoning,
        }

    # -----------------------------------------------------------
    # Run Evaluation
    # -----------------------------------------------------------

    def evaluate( self, experiment_prefix: str = "RAG_Eval_Langsmith_Native", max_concurrency: int = 2,):
        """
        Run the complete RAG evaluation.
        """

        results = evaluate(
            self.rag_target,
            data=self.dataset_name,
            evaluators=self.evaluators,
            experiment_prefix=experiment_prefix,
            client=self.client,
            max_concurrency=max_concurrency,
        )

        print("Evaluation completed.")
        print(f"Results: {results}")

        return results