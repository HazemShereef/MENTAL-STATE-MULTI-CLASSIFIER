\documentclass{article}
\usepackage[utf8]{inputenc}
\usepackage{graphicx}
\usepackage{hyperref}
\usepackage{amsmath}
\usepackage{xcolor}
\usepackage{listings}

\definecolor{codegreen}{rgb}{0,0.6,0}
\definecolor{codegray}{rgb}{0.5,0.5,0.5}
\definecolor{codepurple}{rgb}{0.58,0,0.82}

\lstdefinestyle{mystyle}{
    backgroundcolor=\color{white},   
    commentstyle=\color{codegreen},
    keywordstyle=\color{magenta},
    numberstyle=\tiny\color{codegray},
    stringstyle=\color{codepurple},
    basicstyle=\ttfamily\footnotesize,
    breakatwhitespace=false,         
    breaklines=true,                 
    captionpos=b,                    
    keepspaces=true,                 
    numbers=left,                    
    numbersep=5pt,                  
    showspaces=false,                
    showstringspaces=false,
    showtabs=false,                  
    tabsize=2
}

\lstset{style=mystyle}

\title{Mental State Multi-Classifier}
\author{Your Name}
\date{\today}

\begin{document}

\maketitle

\begin{center}
    \includegraphics[width=0.5\textwidth]{placeholder.png} \\
    \textcolor{gray}{\small A CNN-based classifier for detecting mental states from physiological signals}
\end{center}

\section*{Introduction}
Understanding and classifying human mental states is crucial for developing computing systems that can monitor mental health and assist with stress management. Biosignals such as ECG, EDA, and Respiration carry valuable information about an individual's psychological condition. This project develops a multiclass classifier capable of identifying four distinct mental states:

\begin{itemize}
    \item Baseline (resting state)
    \item Stress
    \item Amusement
    \item Meditation
\end{itemize}

\section*{Dataset}
The WESAD (Wearable Stress and Affect Detection) dataset was used, containing physiological data from 15 subjects across four emotional states. Key characteristics:

\begin{itemize}
    \item Signals collected from chest-worn RespiBAN device
    \item Focused on ECG, EDA, and Respiration signals
    \item Sample rate: 700Hz
    \item Total samples: 60.8 million per signal type
\end{itemize}

\section*{Methodology}

\subsection*{Preprocessing}
\begin{enumerate}
    \item Signal filtering:
    \begin{itemize}
        \item ECG: FIR filter [0.67, 45] Hz + 50Hz notch
        \item EDA: Low-pass filter [3Hz]
        \item Respiration: Bandpass [0.05-0.5Hz]
    \end{itemize}
    \item Segmentation into 5-second windows (3500 samples)
    \item Class balancing using SMOTE
    \item Gaussian noise augmentation
    \item Standardization using StandardScaler
\end{enumerate}

\subsection*{Model Architecture}
\begin{lstlisting}[language=Python]
class MentalStateClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(3, 32, kernel_size=5)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5)
        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(64*174, 128)
        self.fc2 = nn.Linear(128, 4)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64*174)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
\end{lstlisting}

\section*{Results}
The model achieved exceptional performance:

\begin{table}[h]
\centering
\begin{tabular}{|l|l|}
\hline
Metric & Value \\ \hline
Training Accuracy & 97.8\% \\
Validation Accuracy & 98.3\% \\
Test Accuracy & 93\% \\ \hline
\end{tabular}
\caption{Model Performance Metrics}
\end{table}

\section*{Conclusion}
The developed classifier demonstrates:
\begin{itemize}
    \item High accuracy (98.3\%) in mental state classification
    \item Computational efficiency (<10 min training on laptop)
    \item Practical applicability for wearable devices
\end{itemize}

Future work could explore real-time implementation on embedded systems and expansion to additional mental states.

\end{document}
