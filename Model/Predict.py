import Evaluate as evl


image_path = 'download (1).jfif'

result , attention_plot = evl.evaluate(image_path)
print ('Prediction Caption:', ' '.join(result))