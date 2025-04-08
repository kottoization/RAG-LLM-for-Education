import json
import os
from datetime import date, timedelta, datetime
from langchain_openai import ChatOpenAI

# TODO: use cases from prompts for edu

class LearningPlan:
    def __init__(self, user_name, quiz_results=None, user_goals=None):
        self.user_name = user_name
        self.quiz_results = quiz_results if quiz_results else {}
        self.user_goals = user_goals if user_goals else {}
        self.learning_plan = []
        self.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7, verbose=True)  # Użycie ChatOpenAI

    def analyze_quiz_results(self):
        """
        Analyze quiz results and classify topics based on knowledge gaps.
        """
        critical_areas = []  # Score below 50%
        moderate_areas = []  # Score between 50% and 70%
        good_areas = []      # Score above 70%

        for topic, (correct_answers, total_questions) in self.quiz_results.items():
            if total_questions == 0:  # Uniknięcie dzielenia przez zero
                percentage_score = 0
            else:
                percentage_score = (correct_answers / total_questions) * 100

            if percentage_score < 50:
                critical_areas.append(topic)
            elif 50 <= percentage_score < 70:
                moderate_areas.append(topic)
            else:
                good_areas.append(topic)

        return critical_areas, moderate_areas, good_areas

    def generate_plan(self):
        """
        Generate a learning plan based on quiz analysis.
        """
        critical_areas, moderate_areas, good_areas = self.analyze_quiz_results()
        plan_start_date = date.today()
        plan = []

        # Critical areas (High priority)
        for i, topic in enumerate(critical_areas):
            plan.append({
                'date': plan_start_date + timedelta(days=i * 2),
                'priority': 'High priority',
                'topic': topic,
                'materials': self.recommend_materials(topic)
            })

        # Moderate areas (Medium priority)
        offset = len(critical_areas)
        for i, topic in enumerate(moderate_areas, start=offset):
            plan.append({
                'date': plan_start_date + timedelta(days=i * 2),
                'priority': 'Medium priority',
                'topic': topic,
                'materials': self.recommend_materials(topic)
            })

        # Good areas (Low priority)
        offset += len(moderate_areas)
        for i, topic in enumerate(good_areas, start=offset):
            plan.append({
                'date': plan_start_date + timedelta(days=i * 2),
                'priority': 'Low priority',
                'topic': topic,
                'materials': self.recommend_materials(topic)
            })

        self.learning_plan = plan
        return plan

    def recommend_materials(self, topic):
        """
        Retrieve recommended materials for a given topic using LLM.
        """
        prompt = (
            f"You are an AI assistant tasked with recommending study materials. "
            f"Provide a concise but short list of materials to help someone learn about '{topic}'."
        )
        try:
            # Invoke ChatOpenAI directly with a string prompt
            response = self.llm.invoke(prompt)
            materials = response.content.split("\n")  # Assuming each material is in a new line
            return materials
        except Exception as e:
            print(f"Error while generating materials for topic '{topic}': {e}")
            return ["No materials available"]

    def generate_plan_from_prompt(self, user_input):
        """
        Generate a learning plan based on user's custom input.
        """
        plan_start_date = date.today()
        plan = []

        goals = user_input.get("goals", [])
        for i, goal in enumerate(goals):
            plan.append({
                'date': plan_start_date + timedelta(days=i * 2),
                'priority': 'User-defined',
                'topic': goal,
                'materials': self.recommend_materials(goal)
            })

        self.learning_plan = plan
        return plan

    def display_plan(self):
        """
        Display the generated learning plan.
        """
        print(f"Learning Plan for {self.user_name}:\n")
        for entry in self.learning_plan:
            print(f"Date: {entry['date']}")
            print(f"Topic: {entry['topic']}")
            print(f"Priority: {entry['priority']}")
            print("Recommended Materials:")
            for material in entry['materials']:
                print(f" - {material}")
            print("\n")

    def save_to_file(self, base_dir="data/learning_plans/"):
        """
        Save the generated learning plan to a JSON file with timestamp and user name.
        """
        os.makedirs(base_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.user_name}_plan_{timestamp}.json"
        path = os.path.join(base_dir, filename)

        try:
            plan_serializable = [
                {
                    **entry,
                    "date": entry["date"].isoformat()
                }
                for entry in self.learning_plan
            ]

            with open(path, "w", encoding="utf-8") as f:
                json.dump(plan_serializable, f, indent=4, ensure_ascii=False)

            print(f"✅ Plan saved to {path}")
        except Exception as e:
            print(f"❌ Error saving plan: {e}")