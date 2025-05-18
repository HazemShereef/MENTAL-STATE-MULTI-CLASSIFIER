\documentclass{article}
\usepackage[utf8]{inputenc}
\usepackage{graphicx}
\usepackage{hyperref}
\usepackage{amsmath}

\title{Mental State Multi-Class Classifier}
\author{}
\date{}

\begin{document}

\maketitle

\section*{Introduction}

Understanding and classifying human mental states is crucial for building systems that aid in stress management and mental health monitoring. This project aims to classify four mental states—\textbf{baseline}, \textbf{stress}, \textbf{amusement}, and \textbf{meditation}—using biosignals such as ECG, EDA, and Respiration collected from wearable sensors.

\section*{Dataset Description}

The dataset used is \textbf{WESAD (Wearable Stress and Affect Detection)}, which includes physiological data from 15 subjects under 4 emotional states. We used data from the \textit{RespiBAN} chest-worn sensor for its superior signal quality. The selected biosignals were:
\begin{itemize}
    \item ECG
    \item EDA
    \item Respiration
\end{itemize}

\subsection*{Why these signals?}
\begin{itemize}
    \item \textbf{Quality:} Less noise and motion artifacts.
    \item \textbf{Relevance:} Directly linked to psychological states.
    \item \textbf{Simplicity:} Fewer signals reduce complexity and resource demands.
\end{itemize}

\section*{Data Preprocessing}

\begin{itemize}
    \item Loaded and concatenated raw signals from all subjects.
    \item Filtered samples to include only the four target mental states.
    \item Applied signal cleaning using \textbf{NeuroKit2}:
    \begin{itemize}
        \item ECG: FIR filter \([0.67, 45]\) Hz + notch at 50 Hz
        \item EDA: Low-pass filter at 3 Hz
        \item Respiration: Bandpass filter \([0.05, 0.5]\) Hz
    \end{itemize}
    \item Segmented signals into 5-second windows (3500 samples at 700 Hz).
    \item Data split using \textbf{GroupShuffleSplit} to ensure subject-independent evaluation.
    \item \textbf{Class imbalance} was handled using \textbf{SMOTE}.
    \item \textbf{Gaussian noise} was added for augmentation.
    \item Features standardized using \textbf{StandardScaler}.
\end{itemize}

\section*{Model Architecture}

The classifier is a \textbf{1D CNN} designed to process multi-channel biosignals:
\begin{itemize}
    \item Input: (3 channels $\times$ 3500 samples)
    \item Convolutional layers with \textbf{ReLU} activations
    \item \textbf{Max-pooling} layers for dimensionality reduction
    \item \textbf{Fully connected layers} leading to a linear output layer with 4 units
    \item Loss function: \texttt{CrossEntropyLoss} (PyTorch)
\end{itemize}

Model visualized using TensorBoard with dummy input \((1, 3, 3500)\).

\section*{Training Details}

\begin{itemize}
    \item Optimizer: \textbf{Adam}
    \item Initial learning rate: \texttt{0.001}, dynamically adjusted using scheduler
    \item Epochs: 60
    \item Dropout used to prevent overfitting
    \item Training, validation, and test were strictly subject-split to avoid leakage
\end{itemize}

\subsection*{Overfitting Mitigation}
To overcome early signs of overfitting:
\begin{itemize}
    \item Applied more aggressive dropout
    \item Used SMOTE and data augmentation
\end{itemize}

\section*{Results}

\begin{itemize}
    \item Number of 5-second windows: 39448
    \item Training time: $\sim$10 minutes on a local laptop
    \item Validation accuracy: \textbf{98.3\%}
    \item Test accuracy: \textbf{97.8\%}
\end{itemize}

\section*{Conclusion}

The project demonstrates the feasibility of real-time, multi-class mental state classification using only three biosignals and a lightweight CNN. The approach is efficient and suitable for wearable deployment.

\section*{Acknowledgments}

This project uses data from the \textbf{WESAD} dataset provided by UZ Zurich.

\end{document}
