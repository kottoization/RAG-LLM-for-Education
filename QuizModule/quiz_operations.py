from tools.quiz_prompts import (
    generate_topic_list_prompt,
    generate_questions_prompt,
)
from langchain_openai import ChatOpenAI
# from langchain_core.runnables import RunnableParallel, RunnableLambda
from langchain.schema.runnable import RunnableParallel, RunnableLambda

from LearningPlanModule.learning_plan import LearningPlan

def generate_quiz(subject: str):
    """
    Generates a quiz based on the provided subject using parallel chains.
    """
    try:
        # Initialize the ChatOpenAI model
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.1, verbose=True)

        # Generate topics
        print(f"Generating topics for subject: {subject}")
        topic_prompt = generate_topic_list_prompt(subject)
        try:
            topic_result = llm.invoke(topic_prompt.format_prompt(subject=subject))
            topics = topic_result.content.split("\n")
            print(f"Generated topics: {topics}")
        except Exception as e:
            raise ValueError(f"Error generating topics: {e}")

        # Skip empty topics
        topics = [topic.strip() for topic in topics if topic.strip()]

        # Limit the number of topics and questions
        max_questions = 20
        max_topics = min(len(topics), 5)
        topics = topics[:max_topics]
        questions_per_topic = max_questions // max_topics

        # Generate questions for all topics in parallel
        print("Generating questions for all topics...")
        try:
            question_chain = RunnableLambda(
                lambda inputs: generate_questions_prompt(inputs["topic"]).format_prompt(topic=inputs["topic"])
            ) | llm

            questions = question_chain.batch([{"topic": topic} for topic in topics])
        except Exception as e:
            raise ValueError(f"Error generating questions: {e}")

        # Assemble the quiz interactively
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

                    # Extract correct answer
                    correct_answer = question.split("Correct Answer: ")[-1].strip().lower()
                    if user_answer == correct_answer:
                        print("Correct!\n")
                        correct_answers += 1
                    else:
                        print(f"Wrong! The correct answer is: {correct_answer}\n")
                except Exception as e:
                    print(f"Error parsing question or correct answer: {e}")

            user_scores[topic] = (correct_answers, total_topic_questions)
            total_correct += correct_answers
            total_questions += total_topic_questions

        # Display final results
        print("\nFinal Results:")
        overall_percentage = 0
        for topic, (correct, total) in user_scores.items():
            percentage = (correct / total) * 100 if total > 0 else 1
            overall_percentage += percentage
            print(f"Topic: {topic} - Score: {correct}/{total} ({percentage:.2f}%)")

        # Calculate and display the overall score
        overall_percentage /= len(user_scores)
        print(f"\nOverall Score: {total_correct}/{total_questions} ({overall_percentage:.2f}%)")

        return user_scores

    except Exception as e:
        print(f"An error occurred while generating the quiz: {e}")
        return {}

def generate_learning_plan_from_quiz(user_name, quiz_results):
    """
    Generates a learning plan based on quiz results.
    """
    plan = LearningPlan(user_name=user_name, quiz_results=quiz_results)
    learning_plan = plan.generate_plan()
    plan.display_plan()
    return learning_plan
