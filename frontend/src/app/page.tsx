'use client'

import { useState, useCallback, useRef } from 'react'
import axios from 'axios'
import { Upload, Loader2, CheckCircle, AlertCircle, Image as ImageIcon } from 'lucide-react'

interface PredictionResult {
  class: string
  confidence: number
}

export default function Home() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [previewUrl, setPreviewUrl] = useState<string | null>(null)
  const [result, setResult] = useState<PredictionResult | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [isDragging, setIsDragging] = useState(false)
  const fileInputRef = useRef<HTMLInputElement>(null)

  // Format disease name: Tomato___Bacterial_spot -> Bacterial Spot
  const formatDiseaseName = (className: string): string => {
    // Remove Tomato___ prefix
    let formatted = className.replace(/^Tomato___/, '')
    // Convert underscores to spaces
    formatted = formatted.replace(/_/g, ' ')
    // Convert to title case
    formatted = formatted
      .split(' ')
      .map(word => word.charAt(0).toUpperCase() + word.slice(1).toLowerCase())
      .join(' ')
    return formatted
  }

  // Handle file selection
  const handleFileSelect = useCallback((file: File) => {
    // Validate file type
    if (!file.type.startsWith('image/')) {
      setError('Please select a valid image file')
      return
    }

    setSelectedFile(file)
    setError(null)
    setResult(null)

    // Create preview URL
    const url = URL.createObjectURL(file)
    setPreviewUrl(url)

    // Automatically submit to API
    handleSubmit(file)
  }, [])

  // Handle drag and drop
  const handleDragOver = useCallback((e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault()
    e.stopPropagation()
    setIsDragging(true)
  }, [])

  const handleDragLeave = useCallback((e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault()
    e.stopPropagation()
    setIsDragging(false)
  }, [])

  const handleDrop = useCallback((e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault()
    e.stopPropagation()
    setIsDragging(false)

    const files = e.dataTransfer.files
    if (files && files.length > 0) {
      handleFileSelect(files[0])
    }
  }, [handleFileSelect])

  // Handle file input change
  const handleFileInputChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files
    if (files && files.length > 0) {
      handleFileSelect(files[0])
    }
  }, [handleFileSelect])

  // Submit file to API
  const handleSubmit = async (file: File) => {
    setLoading(true)
    setError(null)

    try {
      const formData = new FormData()
      formData.append('file', file)

      const response = await axios.post<PredictionResult>(
        'http://localhost:8000/predict',
        formData,
        {
          headers: {
            'Content-Type': 'multipart/form-data',
          },
        }
      )

      setResult(response.data)
    } catch (err) {
      if (axios.isAxiosError(err)) {
        if (err.response) {
          setError(err.response.data?.detail || 'Failed to process image')
        } else if (err.request) {
          setError('Unable to connect to the server. Please make sure the backend is running.')
        } else {
          setError('An error occurred while processing your request')
        }
      } else {
        setError('An unexpected error occurred')
      }
    } finally {
      setLoading(false)
    }
  }

  // Reset state
  const handleReset = () => {
    setSelectedFile(null)
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl)
    }
    setPreviewUrl(null)
    setResult(null)
    setError(null)
    if (fileInputRef.current) {
      fileInputRef.current.value = ''
    }
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-emerald-50 to-green-50 dark:from-gray-900 dark:to-gray-800">
      <main className="container mx-auto px-4 py-8 md:py-16 max-w-4xl">
        {/* Header */}
        <div className="text-center mb-8 md:mb-12">
          <h1 className="text-4xl md:text-5xl font-bold text-emerald-700 dark:text-emerald-400 mb-3">
            LeafLens AI
          </h1>
          <p className="text-lg text-gray-600 dark:text-gray-300">
            Plant Disease Detection Powered by Machine Learning
          </p>
        </div>

        {/* Upload Zone */}
        <div className="mb-8">
          <div
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current?.click()}
            className={`
              relative border-2 border-dashed rounded-xl p-8 md:p-12
              cursor-pointer transition-all duration-200
              ${isDragging
                ? 'border-emerald-500 bg-emerald-100 dark:bg-emerald-900/20'
                : 'border-emerald-300 hover:border-emerald-400 bg-white dark:bg-gray-800'
              }
              ${previewUrl ? 'hidden' : ''}
            `}
          >
            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              onChange={handleFileInputChange}
              className="hidden"
            />
            <div className="flex flex-col items-center justify-center text-center">
              <Upload
                className={`w-16 h-16 mb-4 ${isDragging ? 'text-emerald-600' : 'text-emerald-500'}`}
              />
              <p className="text-lg font-semibold text-gray-700 dark:text-gray-200 mb-2">
                {isDragging ? 'Drop your image here' : 'Drag & drop your image here'}
              </p>
              <p className="text-sm text-gray-500 dark:text-gray-400">
                or click to browse
              </p>
            </div>
          </div>

          {/* Image Preview */}
          {previewUrl && (
            <div className="relative bg-white dark:bg-gray-800 rounded-xl p-4 shadow-lg">
              <div className="relative w-full aspect-square max-w-md mx-auto mb-4">
                <img
                  src={previewUrl}
                  alt="Preview"
                  className="w-full h-full object-contain rounded-lg"
                />
              </div>
              <div className="flex gap-3 justify-center">
                <button
                  onClick={handleReset}
                  className="px-4 py-2 text-sm font-medium text-gray-700 dark:text-gray-300 bg-gray-100 dark:bg-gray-700 rounded-lg hover:bg-gray-200 dark:hover:bg-gray-600 transition-colors"
                >
                  Choose Different Image
                </button>
              </div>
            </div>
          )}
        </div>

        {/* Loading State */}
        {loading && (
          <div className="bg-white dark:bg-gray-800 rounded-xl p-6 shadow-lg mb-8">
            <div className="flex items-center justify-center gap-3">
              <Loader2 className="w-6 h-6 text-emerald-600 animate-spin" />
              <p className="text-lg font-medium text-gray-700 dark:text-gray-200">
                Analyzing image...
              </p>
            </div>
          </div>
        )}

        {/* Error State */}
        {error && (
          <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-xl p-6 shadow-lg mb-8">
            <div className="flex items-start gap-3">
              <AlertCircle className="w-6 h-6 text-red-600 dark:text-red-400 flex-shrink-0 mt-0.5" />
              <div>
                <h3 className="font-semibold text-red-800 dark:text-red-300 mb-1">
                  Error
                </h3>
                <p className="text-red-700 dark:text-red-400">{error}</p>
              </div>
            </div>
          </div>
        )}

        {/* Results */}
        {result && !loading && (
          <div className="bg-white dark:bg-gray-800 rounded-xl p-6 md:p-8 shadow-lg">
            <div className="flex items-start gap-3 mb-6">
              <CheckCircle className="w-6 h-6 text-emerald-600 dark:text-emerald-400 flex-shrink-0 mt-0.5" />
              <div className="flex-1">
                <h2 className="text-2xl font-bold text-gray-800 dark:text-gray-100 mb-2">
                  Detection Result
                </h2>
                <div className="mb-4">
                  <p className="text-sm text-gray-500 dark:text-gray-400 mb-1">
                    Disease Name
                  </p>
                  <p className="text-xl font-semibold text-emerald-700 dark:text-emerald-400">
                    {formatDiseaseName(result.class)}
                  </p>
                </div>
                <div>
                  <div className="flex justify-between items-center mb-2">
                    <p className="text-sm text-gray-500 dark:text-gray-400">
                      Confidence
                    </p>
                    <p className="text-sm font-semibold text-gray-700 dark:text-gray-300">
                      {(result.confidence * 100).toFixed(1)}%
                    </p>
                  </div>
                  <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-4 overflow-hidden">
                    <div
                      className="h-full bg-gradient-to-r from-emerald-500 to-emerald-600 transition-all duration-500 ease-out"
                      style={{ width: `${result.confidence * 100}%` }}
                    />
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}
      </main>
    </div>
  )
}
