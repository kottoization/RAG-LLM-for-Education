# QuizModule
# ----------
# This module provides functionality for generating subject-based quizzes
# and corresponding learning plans based on user performance.

from tools.quiz_prompts import generate_topic_list_prompt, generate_questions_prompt
from langchain_openai import ChatOpenAI
from langchain.schema.runnable import RunnableLambda
from LearningPlanModule.learning_plan import LearningPlan
from tools.language_handler import LanguageHandler

def generate_quiz(subject: str, language: str = "en", retriever=None):
    """
    Generates a quiz based on the provided subject using parallel chains.
    If retriever is provided, fetches relevant documents for context.
    """
    try:
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.1, verbose=True)

        # Optional RAG context fetch
        context = ""
        if retriever:
            docs = retriever.get_relevant_documents(subject)
            # concatenate top-k documents
            context = "\n\n".join([doc.page_content for doc in docs])
            print(f"[RAG] Retrieved {len(docs)} documents for context.")

        # Prepare input for topic prompt (include context if available)
        prompt_subject = subject
        if context:
            prompt_subject = f"{subject}\n\nContext:\n{context}"

        # Generate topics
        print(f"Generating topics for subject: {subject}")
        topic_prompt = generate_topic_list_prompt(prompt_subject, language)
        try:
            topic_result = llm.invoke(topic_prompt.format_prompt(subject=prompt_subject))
            topics = topic_result.content.split("\n")
            print(f"Generated topics: {topics}")
        except Exception as e:
            raise ValueError(f"Error generating topics: {e}")

        topics = [t.strip() for t in topics if t.strip()]
        max_questions = 20
        max_topics = min(len(topics), 5)
        topics = topics[:max_topics]
        questions_per_topic = max_questions // max_topics

        # Generate questions in parallel
        print("Generating questions for all topics...")
        try:
            question_chain = RunnableLambda(
                lambda inputs: generate_questions_prompt(inputs["topic"], language=language).format_prompt(topic=inputs["topic"])
            ) | llm
            questions = question_chain.batch([{"topic": topic} for topic in topics])
        except Exception as e:
            raise ValueError(f"Error generating questions: {e}")

        # Quiz CLI
        print("\nStarting the quiz...\n")
        user_scores = {}
        total_questions = 0
        total_correct = 0

        for topic, question_set in zip(topics, questions):
            print(f"Topic: {topic}\n")
            question_texts = question_set.content.split("\n\n")[:questions_per_topic]
            correct_answers = 0
            total_topic_questions = len(question_texts)

            for question in question_texts:
                try:
                    print(question)
                    user_answer = input("Your answer: ").strip().lower()
                    raw_correct = question.split("Correct Answer: ")[-1].strip().lower()
                    correct_answer = raw_correct[0] if raw_correct and raw_correct[0] in ['a','b','c','d'] else "?"
                    if user_answer == correct_answer:
                        print("Correct!\n")
                        correct_answers += 1
                    elif correct_answer == "?":
                        # when missing correct key, count as correct temporarily
                        correct_answers += 1
                    else:
                        print(f"Wrong! The correct answer is: {correct_answer}\n")
                except Exception as e:
                    print(f"Error parsing question or answer: {e}")

            user_scores[topic] = (correct_answers, total_topic_questions)
            total_correct += correct_answers
            total_questions += total_topic_questions

        # Display results
        print("\nFinal Results:")
        overall_percentage = 0
        for topic, (correct, total) in user_scores.items():
            percentage = (correct/total)*100 if total>0 else 0
            overall_percentage += percentage
            print(f"Topic: {topic} - Score: {correct}/{total} ({percentage:.2f}%)")
        overall_percentage /= len(user_scores) if user_scores else 1
        print(f"\nOverall Score: {total_correct}/{total_questions} ({overall_percentage:.2f}%)")

        return user_scores

    except Exception as e:
        print(f"An error occurred while generating the quiz: {e}")
        return {}