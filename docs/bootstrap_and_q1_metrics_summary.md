# Bootstrap and final Q1 metrics summary

All bootstrap estimates used 1,000 paired bootstrap resamples and the full fixed 38-class label space.

## Best adapted model
ResNet50 fine-tuned on PlantDoc with data augmentation achieved:

- Accuracy: 0.525424
- Macro precision: 0.405110
- Macro recall: 0.373055
- Macro F1-score: 0.364831
- Weighted precision: 0.573142
- Weighted recall: 0.525424
- Weighted F1-score: 0.517058

## Bootstrap confidence intervals

- No adaptation: accuracy 0.176992 [0.144068, 0.211864]
- Fine-tuning: accuracy 0.491091 [0.447034, 0.536017]
- Fine-tuning + data augmentation: accuracy 0.525150 [0.480879, 0.569968]

## Gain analysis

- Fine-tuning vs no adaptation: accuracy gain 0.314100 [0.264831, 0.364407]
- Fine-tuning + augmentation vs no adaptation: accuracy gain 0.348159 [0.296610, 0.398305]
- Fine-tuning + augmentation vs fine-tuning: accuracy gain 0.034059 [0.002066, 0.065678]

The accuracy gain from augmentation over simple fine-tuning is positive but modest. Macro-F1 and weighted-F1 gains over simple fine-tuning include zero in their 95% confidence intervals and should be interpreted cautiously.
