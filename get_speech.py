import azure.cognitiveservices.speech as speechsdk

import requests
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO


def get_text():
    # Creates an instance of a speech config with specified subscription key and service region.
    # Replace with your own subscription key and service region (e.g., "westus").
    speech_key, service_region = "02773ba094fa4c55be7510e2e317d9be", "westus"
    speech_config = speechsdk.SpeechConfig(
        subscription=speech_key, region=service_region)

    # Creates a recognizer with the given settings
    speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config)

    print("Say something...")

    # Starts speech recognition, and returns after a single utterance is recognized. The end of a
    # single utterance is determined by listening for silence at the end or until a maximum of 15
    # seconds of audio is processed.  The task returns the recognition text as result.
    # Note: Since recognize_once() returns only a single utterance, it is suitable only for single
    # shot recognition like command or query.
    # For long-running multi-utterance recognition, use start_continuous_recognition() instead.
    result = speech_recognizer.recognize_once()

    # Checks result.
    if result.reason == speechsdk.ResultReason.RecognizedSpeech:
        print("Recognized: {}".format(result.text))
        return result.text
    elif result.reason == speechsdk.ResultReason.NoMatch:
        print("No speech could be recognized: {}".format(result.no_match_details))
    elif result.reason == speechsdk.ResultReason.Canceled:
        cancellation_details = result.cancellation_details
        print("Speech Recognition canceled: {}".format(
            cancellation_details.reason))
        if cancellation_details.reason == speechsdk.CancellationReason.Error:
            print("Error details: {}".format(
                cancellation_details.error_details))


subscription_key = "21b5911a1e084a9d8aa92127502e072f"
search_url = "https://eastus.api.cognitive.microsoft.com/bing/v7.0/images/search"
search_term = get_text()

terms = search_term.split(' ')

imgs = []
for term in terms:
    print(term)
    headers = {"Ocp-Apim-Subscription-Key": subscription_key}
    params = {"q": term, "license": "public", "imageType": "photo"}
    response = requests.get(search_url, headers=headers, params=params)
    response.raise_for_status()
    search_results = response.json()
    imgs.append(search_results['value'][0]['thumbnailUrl'])

f, axes = plt.subplots(len(imgs))
for i, imgLink in enumerate(imgs):
    image_data = requests.get(imgLink)
    image_data.raise_for_status()
    image = Image.open(BytesIO(image_data.content))
    axes[i].imshow(image)
    axes[i].axis("off")

plt.show()
