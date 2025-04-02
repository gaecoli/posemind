'use client';

import { useState } from 'react';

export default function Home() {
  const [video, setVideo] = useState<File | null>(null);
  const [analysis, setAnalysis] = useState<string>('');
  const [loading, setLoading] = useState(false);

  const handleVideoUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setVideo(e.target.files[0]);
      setAnalysis('');
    }
  };

  const handleAnalyze = async () => {
    if (!video) return;
    
    setLoading(true);
    const formData = new FormData();
    formData.append('video', video);

    try {
      const response = await fetch('http://localhost:8000/analyze/pushup', {
        method: 'POST',
        body: formData,
      });
      
      const data = await response.json();
      setAnalysis(data.message);
    } catch (error) {
      setAnalysis('分析过程中出现错误，请重试');
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="min-h-screen bg-gradient-to-b from-gray-900 to-gray-800 text-white">
      <div className="container mx-auto px-4 py-16">
        <div className="text-center">
          <h1 className="text-5xl font-bold mb-6">PoseMind</h1>
          <p className="text-xl text-gray-300 mb-12">
            AI 驱动的动作分析系统
          </p>
          
          <div className="max-w-2xl mx-auto bg-gray-800 p-8 rounded-lg">
            <h2 className="text-2xl font-semibold mb-6">俯卧撑动作分析</h2>
            
            <div className="mb-6">
              <input
                type="file"
                accept="video/*"
                onChange={handleVideoUpload}
                className="block w-full text-sm text-gray-300
                  file:mr-4 file:py-2 file:px-4
                  file:rounded-full file:border-0
                  file:text-sm file:font-semibold
                  file:bg-blue-500 file:text-white
                  hover:file:bg-blue-600"
              />
            </div>

            {video && (
              <div className="mb-6">
                <video
                  src={URL.createObjectURL(video)}
                  controls
                  className="w-full rounded-lg"
                />
              </div>
            )}

            <button
              onClick={handleAnalyze}
              disabled={!video || loading}
              className="bg-blue-500 hover:bg-blue-600 text-white font-bold py-2 px-4 rounded
                disabled:bg-gray-500 disabled:cursor-not-allowed"
            >
              {loading ? '分析中...' : '开始分析'}
            </button>

            {analysis && (
              <div className="mt-6 p-4 bg-gray-700 rounded-lg">
                <h3 className="text-xl font-semibold mb-2">分析结果：</h3>
                <p className="text-gray-300">{analysis}</p>
              </div>
            )}
          </div>
        </div>
      </div>
    </main>
  );
} 