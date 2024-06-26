import numpy as np
from typing import Dict, List, Union, Tuple
from pydub import AudioSegment
import pandas as pd
import torch
import seaborn as sns
from explainers.explanation_speech import ExplanationSpeech
from transcribe_functions import pydub_to_np, print_log

SCORES_PALETTE = sns.diverging_palette(240, 10, as_cmap=True)
from IPython.display import display

from explainers.loo_speech_explainer import LOOSpeechExplainer
from explainers.lime_speech_explainer import LIMESpeechExplainer
from explainers.paraling_speech_explainer import ParalinguisticSpeechExplainer

## Set seed
SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)
# If True, We use the add_noise of torch audio
USE_ADD_NOISE_TORCHAUDIO = True

REFERENCE_STR = "-"


class Benchmark:
    def __init__(
        self,
        model,
        feature_extractor,
        device: str = "cuda:0",
        language: str = "en",
        explainers=None,
    ):
        self.model = model
        self.feature_extractor = feature_extractor
        self.model.eval()
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        if "superb-ic" in self.model.name_or_path:
            # We are using the FSC model - It has three output classes
            from helpers.model_helper_fsc import ModelHelperFSC

            self.model_helper = ModelHelperFSC(
                self.model, self.feature_extractor, self.device, "en"
            )

        if explainers is None:
            # Use the default explainers
            self.explainers = {
                "LOO": LOOSpeechExplainer(self.model_helper),
                "LIME": LIMESpeechExplainer(self.model_helper),
                "perturb_paraling": ParalinguisticSpeechExplainer(self.model_helper),
            }

    def set_explainers(self, explainers):
        self.explainers = explainers

    def predict(
        self,
        audios: List[np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # TMP
        # Just a wrapper around ModelHelperFSC.predict/ModelHelperFSC.predict_single We use the second to overcome the padding issue
        return self.model_helper.predict(audios)

    def explain(
        self,
        audio_path: str,
        target_class: str = None,
        methodology: str = "LOO",
        perturbation_types: List[str] = [
            "pitch shifting",
            "pitch shifting down",
            "pitch shifting up",
            "time stretching",
            "time stretching down",
            "time stretching up",
            "reverberation",
            "noise",
        ],
        removal_type: str = "silence",  # Used only for LOO and LIME - explainer_args TODO
        aggregation: str = "mean",  # Used only for Gradient and GradientXInput - explainer_args TODO
        num_samples: int = 1000,  # Used only for LIME - explainer_args TODO
        words_trascript: List = None,
        verbose: bool = False,
        verbose_target: int = 0,
    ):
        """
        Explain the prediction of the model.
        Returns the importance of each segment in the audio.
        """
        explainer_args = {}
        # TODO UNIFY THE INPUT FORMAT

        ## Get the importance of each class (action, object, location) according to the perturb_paraling type
        if methodology == "perturb_paraling":
            explanations = []
            explainer = self.explainers["perturb_paraling"]
            for perturbation_type in perturbation_types:
                explanation = explainer.compute_explanation(
                    audio_path=audio_path,
                    target_class=target_class,
                    perturbation_type=perturbation_type,
                    verbose=verbose,
                    verbose_target=verbose_target,
                )
                explanations.append(explanation)

            # table = self.create_table(importances)
        ## Get the importance of each word
        # elif:

        else:
            if methodology not in self.explainers:
                raise ValueError(
                    f'Explainer {methodology} not supported. Choose between "LOO", "Gradient", "GradientXInput", "LIME", "perturb_paraling"'
                )
            if "LOO" in methodology:
                explainer_args["removal_type"] = removal_type
            elif "LIME" in methodology:
                explainer_args["removal_type"] = removal_type
                explainer_args["num_samples"] = num_samples
            else:
                explainer_args["aggregation"] = aggregation

            explainer = self.explainers[methodology]
            explanation = explainer.compute_explanation(
                audio_path=audio_path,
                target_class=target_class,
                words_trascript=words_trascript,
                **explainer_args,
            )
            explanations = explanation

        ## Return table of explanations
        return explanations

    def create_table(
        self,
        explanations,
        axis=1,  # append the scores to the columns
    ) -> pd.DataFrame:
        """
        Args:
            explanations: list of explanations or single explanation
            axis: 0 for appending the scores by rows, 1 for happending by columns
        Creates a table with the words or paralinguistic feature(s),
        and the difference p(y|x\F) - p(y|x) for each class.
        """

        if type(explanations) == list:
            if axis == 1:
                # Append the scores to the columns

                # We have a list of explanations for the same target class, each for a single feature
                assert [
                    False
                    for i in range(0, len(explanations) - 1)
                    if explanations[i].target != explanations[i + 1].target
                ] == [], "The explanations must have the same target class"
                assert [
                    True
                    for explanation in explanations
                    if len(explanation.features) > 1
                ] == [], "The explanation feature should only be one"
                importance_df = pd.DataFrame(
                    [explanation.scores for explanation in explanations]
                ).T
                importance_df.columns = [
                    explanation.features[0] for explanation in explanations
                ]

                # We take the target of the first explanation. We know that all the explanations have the same target
                target_class_names = self.model_helper.get_text_labels_with_class(
                    explanations[0].target
                )
                importance_df.index = (
                    list(target_class_names)
                    if self.model_helper.n_labels > 1  # multilabel
                    else [target_class_names]  # single label
                )

            else:
                raise ValueError("TODO ")

        else:
            explanation = explanations
            importance_df = pd.DataFrame(explanation.scores)
            importance_df.columns = explanation.features
            target_class_names = self.model_helper.get_text_labels_with_class(
                explanation.target
            )
            importance_df.index = (
                list(target_class_names)
                if self.model_helper.n_labels > 1  # multilabel
                else [target_class_names]  # single label
            )
        return importance_df

    def show_table(self, explanations, apply_style: bool = True, decimals=4):
        # Rename duplicate columns (tokens) by adding a suffix
        table = self.create_table(explanations)

        if sum(table.columns.duplicated().astype(int)) > 0:
            table.columns = pd.io.parsers.base_parser.ParserBase(
                {"names": table.columns, "usecols": None}
            )._maybe_dedup_names(table.columns)

        return (
            table.apply(pd.to_numeric)
            .style.background_gradient(axis=1, cmap=SCORES_PALETTE, vmin=-1, vmax=1)
            .format(precision=decimals)
            if apply_style
            else table.apply(pd.to_numeric).style.format(precision=decimals)
        )

    def explain_variations(self, audio_path, perturbation_types, target_class=None):
        perturbation_df_by_type = self.explainers[
            "perturb_paraling"
        ].explain_variations(audio_path, perturbation_types, target_class)
        return perturbation_df_by_type

    def plot_variations(self, perturbation_df_by_type, show_diff=False, figsize=(5, 5)):
        """
        perturbation_df_by_type: dictionary of dataframe
        show_diff: if True, show the difference with the baseline
        """
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors

        plt.rcParams.update(
            {
                "text.usetex": False,
            }
        )

        fig, axs = plt.subplots(len(perturbation_df_by_type), 1, figsize=figsize)

        for e, (perturbation_type, perturbation_df) in enumerate(
            perturbation_df_by_type.items()
        ):
            ax = axs[e]

            prob_variations_np = perturbation_df.values

            if show_diff:
                reference_value = REFERENCE_STR

                prob_variations_np = (
                    perturbation_df[reference_value].values.reshape(-1, 1)
                    - prob_variations_np
                )

            target_classes_show = list(perturbation_df.index)

            x_labels = list(perturbation_df.columns)
            label_size = 11
            if show_diff:
                norm = mcolors.TwoSlopeNorm(vmin=-1, vmax=1, vcenter=0)
                cmap = plt.cm.PiYG.reversed()
            else:
                norm = mcolors.TwoSlopeNorm(vmin=0, vmax=1, vcenter=0.5)
                cmap = plt.cm.Purples
            im = ax.imshow(prob_variations_np, cmap=cmap, norm=norm, aspect="auto")
            ax.set_yticks(
                np.arange(len(target_classes_show)),
                labels=target_classes_show,
                fontsize=label_size,
            )

            if len(x_labels) > 10:
                if (
                    perturbation_type == "time stretching"
                    or perturbation_type == "noise"
                ):
                    x_labels = [
                        x_label if ((x_label == REFERENCE_STR) or (e % 3 == 0)) else ""
                        for e, x_label in enumerate(x_labels)
                    ]
                else:
                    x_labels = [
                        x_label if ((x_label == REFERENCE_STR) or (e % 4 == 0)) else ""
                        for e, x_label in enumerate(x_labels)
                    ]

            x_labels = [
                r"x" if (x_label == REFERENCE_STR) else x_label for x_label in x_labels
            ]

            if perturbation_type == "time stretching":
                x_labels[0] = str(x_labels[0]) + "\nfaster"
                x_labels[-1] = str(x_labels[-1]) + "\nslower"

                ax.set_xlabel("stretching factor", fontsize=label_size, labelpad=-2)
            elif perturbation_type == "pitch shifting":
                x_labels[0] = str(x_labels[0]) + "\nlower"
                x_labels[-1] = str(x_labels[-1]) + "\nhigher"
                ax.set_xlabel("semitones", fontsize=label_size, labelpad=-2)
            elif perturbation_type == "noise":
                x_labels[-1] = str(x_labels[-1]) + "\nnoiser"
                ax.set_xlabel(
                    "signal-to-noise ratio (dB)", fontsize=label_size, labelpad=-2
                )
            ax.set_xticks(
                np.arange(len(x_labels)), labels=x_labels, fontsize=label_size
            )

            ax.set_title(perturbation_type, fontsize=label_size)

            for lab in ax.get_xticklabels():
                if lab.get_text() == "x":
                    lab.set_fontweight("bold")
        plt.tight_layout()
        plt.subplots_adjust(hspace=2.3)
        cbar = fig.colorbar(
            im,
            ax=axs.ravel().tolist(),
            location="right",
            anchor=(0.3, 0.3),  # 0.3
            shrink=0.7,
        )
        cbar.ax.tick_params(labelsize=label_size)

        plt.show()
        return fig
