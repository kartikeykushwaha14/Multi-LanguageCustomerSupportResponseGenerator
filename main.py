import google.generativeai as genai
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import json
import re
from datetime import datetime
import logging
from collections import defaultdict
import time
from abc import ABC, abstractmethod

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('CustomerSupportGenerator')


class BaseAnalyzer(ABC):
    """Abstract base class for analysis components"""

    @abstractmethod
    def analyze(self, text: str) -> Dict:
        pass


class ComplaintAnalyzer(BaseAnalyzer):
    """Advanced NLP analysis for customer complaints"""

    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-pro')

        # Predefined categories for classification
        self.categories = [
            "Billing Issue", "Technical Problem", "Product Quality",
            "Shipping/Delivery", "Account Management", "General Inquiry",
            "Refund Request", "Feature Request", "Complaint", "Praise"
        ]

        # Sentiment intensity mapping
        self.sentiment_map = {
            "Very Negative": -0.8,
            "Negative": -0.4,
            "Neutral": 0,
            "Positive": 0.4,
            "Very Positive": 0.8
        }

        # Urgency indicators
        self.urgency_indicators = [
            "urgent", "asap", "immediately", "right away", "emergency",
            "critical", "important", "broken", "not working", "failed"
        ]

    def analyze(self, text: str) -> Dict:
        """Comprehensive analysis of customer complaint"""
        try:
            # Sentiment analysis
            sentiment_prompt = f"""
            Analyze the sentiment of this customer message and classify it as one of: 
            Very Negative, Negative, Neutral, Positive, Very Positive.
            Provide only the classification label.

            Message: {text}
            """
            sentiment_response = self.model.generate_content(sentiment_prompt)
            sentiment = sentiment_response.text.strip()

            # Category classification
            category_prompt = f"""
            Classify this customer message into one of these categories: {', '.join(self.categories)}.
            Provide only the category name.

            Message: {text}
            """
            category_response = self.model.generate_content(category_prompt)
            category = category_response.text.strip()

            # Urgency detection
            urgency_score = self._calculate_urgency(text)

            # Key issue extraction
            issue_prompt = f"""
            Extract the main issue or problem described in this customer message.
            Respond with a concise phrase describing the core issue.

            Message: {text}
            """
            issue_response = self.model.generate_content(issue_prompt)
            main_issue = issue_response.text.strip()

            # Emotion detection
            emotion_prompt = f"""
            Identify the primary emotion expressed in this message. Choose from:
            Anger, Frustration, Confusion, Satisfaction, Happiness, Disappointment, 
            Impatience, Gratitude, Anxiety, Other.
            Provide only the emotion label.

            Message: {text}
            """
            emotion_response = self.model.generate_content(emotion_prompt)
            emotion = emotion_response.text.strip()

            return {
                "sentiment": sentiment,
                "sentiment_score": self.sentiment_map.get(sentiment, 0),
                "category": category,
                "urgency_score": urgency_score,
                "main_issue": main_issue,
                "emotion": emotion,
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Error in complaint analysis: {e}")
            return {
                "sentiment": "Neutral",
                "sentiment_score": 0,
                "category": "General Inquiry",
                "urgency_score": 0.5,
                "main_issue": "Unknown",
                "emotion": "Other",
                "timestamp": datetime.now().isoformat()
            }

    def _calculate_urgency(self, text: str) -> float:
        """Calculate urgency score based on text content"""
        text_lower = text.lower()

        # Base score from keyword matching
        keyword_score = 0
        for indicator in self.urgency_indicators:
            if indicator in text_lower:
                keyword_score += 0.1

        # Cap the keyword score
        keyword_score = min(keyword_score, 0.5)

        # Additional analysis for urgency context
        try:
            urgency_prompt = f"""
            Rate the urgency of this customer message on a scale from 0 to 1, where:
            0 = Not urgent at all, 0.5 = Moderately urgent, 1 = Extremely urgent.
            Provide only the numerical score.

            Message: {text}
            """
            urgency_response = self.model.generate_content(urgency_prompt)
            llm_score = float(urgency_response.text.strip())

            # Combine scores (weighted average)
            combined_score = (keyword_score * 0.3) + (llm_score * 0.7)
            return min(combined_score, 1.0)

        except:
            return keyword_score


class ResponseGenerator(BaseAnalyzer):
    """LLM-powered response generation with tone adaptation"""

    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-pro')

        # Tone templates for different scenarios
        self.tone_templates = {
            "apology": "We sincerely apologize for {issue}. We understand this is frustrating and we're working to resolve it.",
            "gratitude": "Thank you for bringing this to our attention. We appreciate your feedback about {issue}.",
            "reassurance": "We want to assure you that we're taking your concern about {issue} seriously and are addressing it.",
            "urgency": "We recognize the urgency of your issue with {issue} and are prioritizing a solution.",
            "neutral": "We've received your message regarding {issue} and are looking into it."
        }

    def generate_response(self, complaint_text: str, analysis: Dict, language: str = "en") -> Dict:
        """Generate a culturally appropriate response based on analysis"""
        try:
            # Determine appropriate tone based on analysis
            tone = self._determine_tone(analysis)

            # Construct prompt for response generation
            prompt = self._construct_prompt(complaint_text, analysis, tone, language)

            # Generate response
            response = self.model.generate_content(prompt)
            response_text = response.text.strip()

            # Calculate predicted satisfaction
            satisfaction_score = self._predict_satisfaction(complaint_text, response_text)

            return {
                "response": response_text,
                "tone": tone,
                "language": language,
                "predicted_satisfaction": satisfaction_score,
                "generated_at": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Error in response generation: {e}")
            return {
                "response": "Thank you for your message. We're looking into your concern and will respond shortly.",
                "tone": "neutral",
                "language": language,
                "predicted_satisfaction": 0.5,
                "generated_at": datetime.now().isoformat()
            }

    def _determine_tone(self, analysis: Dict) -> str:
        """Determine appropriate tone based on sentiment and emotion"""
        sentiment_score = analysis.get("sentiment_score", 0)
        emotion = analysis.get("emotion", "").lower()

        if sentiment_score < -0.5 or emotion in ["anger", "frustration"]:
            return "apology"
        elif sentiment_score > 0.3 or emotion in ["gratitude", "satisfaction"]:
            return "gratitude"
        elif analysis.get("urgency_score", 0) > 0.7:
            return "urgency"
        else:
            return "neutral"

    def _construct_prompt(self, complaint_text: str, analysis: Dict, tone: str, language: str) -> str:
        """Construct detailed prompt for response generation"""
        category = analysis.get("category", "General Inquiry")
        main_issue = analysis.get("main_issue", "the issue")

        prompt = f"""
        You are a customer support representative for a global company.

        CUSTOMER MESSAGE: {complaint_text}

        ANALYSIS:
        - Category: {category}
        - Main Issue: {main_issue}
        - Sentiment: {analysis.get('sentiment', 'Neutral')}
        - Emotion: {analysis.get('emotion', 'Other')}

        Generate a professional, empathetic customer support response in {language}.
        The response should be culturally appropriate for the language.

        GUIDELINES:
        1. Acknowledge the customer's issue clearly
        2. Show empathy and understanding of their situation
        3. Provide a clear course of action or next steps
        4. Maintain a {tone} tone appropriate for the situation
        5. Keep the response concise but thorough (3-5 sentences)
        6. Use appropriate greeting and closing for the language

        RESPONSE (in {language}):
        """

        return prompt

    def _predict_satisfaction(self, complaint: str, response: str) -> float:
        """Predict customer satisfaction score for the response"""
        try:
            prompt = f"""
            Predict how satisfied a customer would be with this support response to their complaint.
            Rate on a scale from 0 to 1, where:
            0 = Very dissatisfied, 0.5 = Neutral, 1 = Very satisfied.
            Provide only the numerical score.

            CUSTOMER COMPLAINT: {complaint}

            SUPPORT RESPONSE: {response}
            """

            satisfaction_response = self.model.generate_content(prompt)
            score = float(satisfaction_response.text.strip())
            return max(0, min(1, score))  # Ensure between 0-1

        except:
            return 0.5  # Default neutral score


class TranslationManager:
    """Handles translation with cultural adaptation"""

    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-pro')

        # Supported languages with cultural notes
        self.supported_languages = {
            "en": "English",
            "es": "Spanish",
            "fr": "French",
            "de": "German",
            "it": "Italian",
            "pt": "Portuguese",
            "ru": "Russian",
            "zh": "Chinese",
            "ja": "Japanese",
            "ko": "Korean",
            "ar": "Arabic",
            "hi": "Hindi",
            # Add more languages as needed
        }

    def translate_with_cultural_adaptation(self, text: str, target_lang: str, context: Dict = None) -> str:
        """Translate text with cultural adaptation"""
        try:
            # Prepare cultural context prompt
            cultural_context = ""
            if context:
                cultural_context = f"Additional context: {json.dumps(context)}"

            prompt = f"""
            Translate the following customer support message to {self.supported_languages[target_lang]} ({target_lang}).
            Ensure the translation is culturally appropriate and maintains the original tone and intent.

            {cultural_context}

            Original text: {text}

            Translation:
            """

            response = self.model.generate_content(prompt)
            return response.text.strip()

        except Exception as e:
            logger.error(f"Translation error: {e}")
            return text  # Return original text on error

    def get_supported_languages(self) -> List[str]:
        """Return list of supported language codes"""
        return list(self.supported_languages.keys())


class QualityMetrics:
    """Tracks and analyzes quality metrics"""

    def __init__(self):
        self.metrics = {
            "total_tickets": 0,
            "by_category": defaultdict(int),
            "by_language": defaultdict(int),
            "avg_satisfaction": 0,
            "avg_response_time": 0,
            "satisfaction_scores": [],
            "response_times": []
        }

        self.start_time = datetime.now()

    def update_metrics(self, analysis: Dict, response: Dict, processing_time: float):
        """Update metrics with new ticket data"""
        self.metrics["total_tickets"] += 1

        # Update category count
        category = analysis.get("category", "Unknown")
        self.metrics["by_category"][category] += 1

        # Update language count
        language = response.get("language", "en")
        self.metrics["by_language"][language] += 1

        # Update satisfaction scores
        satisfaction = response.get("predicted_satisfaction", 0.5)
        self.metrics["satisfaction_scores"].append(satisfaction)
        self.metrics["avg_satisfaction"] = np.mean(self.metrics["satisfaction_scores"])

        # Update response times
        self.metrics["response_times"].append(processing_time)
        self.metrics["avg_response_time"] = np.mean(self.metrics["response_times"])

    def get_report(self) -> Dict:
        """Generate comprehensive metrics report"""
        uptime = (datetime.now() - self.start_time).total_seconds()

        return {
            "uptime_seconds": uptime,
            "total_tickets_processed": self.metrics["total_tickets"],
            "tickets_by_category": dict(self.metrics["by_category"]),
            "tickets_by_language": dict(self.metrics["by_language"]),
            "average_predicted_satisfaction": round(self.metrics["avg_satisfaction"], 3),
            "average_response_time_seconds": round(self.metrics["avg_response_time"], 3),
            "performance_tps": round(self.metrics["total_tickets"] / uptime, 3) if uptime > 0 else 0
        }


class CustomerSupportSystem:
    """Main system orchestrating the customer support pipeline"""

    def __init__(self, gemini_api_key: str):
        self.api_key = gemini_api_key

        # Initialize components
        self.analyzer = ComplaintAnalyzer("AIzaSyAcs7yUquk_FG6Nisn4Ugxfxf_4gtZlKEY")
        self.translation_manager = TranslationManager("AIzaSyAcs7yUquk_FG6Nisn4Ugxfxf_4gtZlKEY")
        self.metrics = QualityMetrics()

        logger.info("Customer Support System initialized")

    def process_ticket(self, complaint_text: str, preferred_language: str = "en") -> Dict:
        """Process a customer support ticket end-to-end"""
        start_time = time.time()

        try:
            # Step 1: Analyze complaint
            analysis = self.analyzer.analyze(complaint_text)
            logger.info(f"Complaint analyzed: {analysis['category']} ({analysis['sentiment']})")

            # Step 2: Generate response in English first
            response = self.response_generator.generate_response(
                complaint_text, analysis, language="en"
            )

            # Step 3: Translate if needed
            if preferred_language != "en" and preferred_language in self.translation_manager.get_supported_languages():
                translated_response = self.translation_manager.translate_with_cultural_adaptation(
                    response["response"],
                    preferred_language,
                    context={"tone": response["tone"], "category": analysis["category"]}
                )
                response["response"] = translated_response
                response["language"] = preferred_language

            # Step 4: Calculate processing time
            processing_time = time.time() - start_time

            # Step 5: Update metrics
            self.metrics.update_metrics(analysis, response, processing_time)

            # Prepare final result
            result = {
                "analysis": analysis,
                "response": response,
                "processing_time_seconds": round(processing_time, 3),
                "success": True
            }

            logger.info(f"Ticket processed successfully in {processing_time:.3f}s")
            return result

        except Exception as e:
            logger.error(f"Error processing ticket: {e}")
            processing_time = time.time() - start_time

            return {
                "analysis": {
                    "sentiment": "Neutral",
                    "sentiment_score": 0,
                    "category": "General Inquiry",
                    "urgency_score": 0.5,
                    "main_issue": "Error occurred",
                    "emotion": "Other",
                    "timestamp": datetime.now().isoformat()
                },
                "response": {
                    "response": "We apologize, but we're experiencing technical difficulties. Please try again later or contact our support team directly.",
                    "tone": "apology",
                    "language": preferred_language,
                    "predicted_satisfaction": 0.3,
                    "generated_at": datetime.now().isoformat()
                },
                "processing_time_seconds": round(processing_time, 3),
                "success": False,
                "error": str(e)
            }

    def get_performance_metrics(self) -> Dict:
        """Get current performance metrics"""
        return self.metrics.get_report()

    def batch_process_tickets(self, tickets: List[str], languages: List[str] = None) -> List[Dict]:
        """Process multiple tickets in batch"""
        if languages is None:
            languages = ["en"] * len(tickets)

        results = []
        for i, (ticket, lang) in enumerate(zip(tickets, languages)):
            logger.info(f"Processing ticket {i + 1}/{len(tickets)}")
            result = self.process_ticket(ticket, lang)
            results.append(result)

            # Add small delay to avoid rate limiting
            time.sleep(0.1)

        return results


# Example usage and demonstration
def demo_system(api_key: str):
    """Demonstrate the customer support system"""
    print("=== Multi-Language Customer Support Generator Demo ===\n")

    # Initialize system
    css = CustomerSupportSystem(api_key)

    # Sample complaints in different scenarios
    sample_complaints = [
        {
            "text": "Your product stopped working after just one week! This is completely unacceptable and I want a full refund immediately.",
            "language": "en"
        },
        {
            "text": "I've been waiting for my package for over two weeks now. The tracking hasn't updated in days and I need this for an important event.",
            "language": "es"
        },
        {
            "text": "The mobile app keeps crashing every time I try to upload a photo. I've already uninstalled and reinstalled it three times!",
            "language": "fr"
        },
        {
            "text": "I just wanted to say thank you for the excellent service I received yesterday. Maria was incredibly helpful and patient with all my questions.",
            "language": "de"
        }
    ]

    # Process each complaint
    for i, complaint in enumerate(sample_complaints, 1):
        print(f"\n--- Processing Complaint #{i} ---")
        print(f"Original: {complaint['text']}")

        result = css.process_ticket(complaint['text'], complaint['language'])

        print(f"Category: {result['analysis']['category']}")
        print(f"Sentiment: {result['analysis']['sentiment']}")
        print(f"Urgency: {result['analysis']['urgency_score']:.2f}")
        print(f"Generated Response ({result['response']['language']}):")
        print(result['response']['response'])
        print(f"Predicted Satisfaction: {result['response']['predicted_satisfaction']:.2f}")
        print(f"Processing Time: {result['processing_time_seconds']:.3f}s")

    # Show performance metrics
    print("\n--- Performance Metrics ---")
    metrics = css.get_performance_metrics()
    for key, value in metrics.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    # Replace with your actual Gemini API key
    GEMINI_API_KEY = "AIzaSyAcs7yUquk_FG6Nisn4Ugxfxf_4gtZlKEY"

    # Run demonstration
    demo_system(GEMINI_API_KEY)