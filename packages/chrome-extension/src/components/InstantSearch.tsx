import { useState, useEffect, useRef } from 'react'
import { Search, X, Calendar, Paperclip } from 'lucide-react'

interface Suggestion {
  type: 'pattern' | 'sender' | 'topic' | 'date' | 'subject'
  text: string
  description: string
}

interface SearchFilters {
  dateRange?: '7d' | '30d' | '90d'
  senders?: string[]
  hasAttachment?: boolean
}

interface SearchResult {
  thread_id: string
  subject: string
  sender: string
  content: string
  created_at: string
  importance_score?: number
  has_attachment?: boolean
}

interface InstantSearchProps {
  onResultsChange?: (results: SearchResult[]) => void
}

export default function InstantSearch({ onResultsChange }: InstantSearchProps) {
  const [searchQuery, setSearchQuery] = useState('')
  const [suggestions, setSuggestions] = useState<Suggestion[]>([])
  const [inlineCompletion, setInlineCompletion] = useState('')
  const [filters, setFilters] = useState<SearchFilters>({})
  const [searchResults, setSearchResults] = useState<SearchResult[]>([])
  const [isSearching, setIsSearching] = useState(false)

  const searchRef = useRef<HTMLDivElement>(null)
  const debounceTimer = useRef<NodeJS.Timeout | null>(null)
  const inputRef = useRef<HTMLInputElement>(null)

  // Update inline completion when suggestions change
  useEffect(() => {
    if (suggestions.length > 0 && searchQuery.length >= 2) {
      const topSuggestion = suggestions[0].text
      // Only show completion if it starts with what user typed
      if (topSuggestion.toLowerCase().startsWith(searchQuery.toLowerCase())) {
        setInlineCompletion(topSuggestion)
      } else {
        setInlineCompletion('')
      }
    } else {
      setInlineCompletion('')
    }
  }, [suggestions, searchQuery])

  // Fetch autocomplete suggestions
  useEffect(() => {
    if (debounceTimer.current) {
      clearTimeout(debounceTimer.current)
    }

    if (searchQuery.length < 2) {
      // Show default suggestions
      fetchSuggestions('')
      return
    }

    debounceTimer.current = setTimeout(() => {
      fetchSuggestions(searchQuery)
    }, 300)

    return () => {
      if (debounceTimer.current) {
        clearTimeout(debounceTimer.current)
      }
    }
  }, [searchQuery])

  const fetchSuggestions = async (query: string) => {
    try {
      const response = await fetch(`http://localhost:8000/autocomplete?query=${encodeURIComponent(query)}`)
      if (response.ok) {
        const data = await response.json()
        setSuggestions(data.suggestions || [])
      }
    } catch (error) {
      console.error('Failed to fetch suggestions:', error)
    }
  }

  const performSearch = async () => {
    if (!searchQuery.trim() && Object.keys(filters).length === 0) {
      setSearchResults([])
      onResultsChange?.([])
      return
    }

    setIsSearching(true)
    try {
      const response = await fetch('http://localhost:8000/instant-search', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: searchQuery,
          filters,
          limit: 20
        }),
        credentials: 'include'
      })

      if (response.ok) {
        const data = await response.json()
        setSearchResults(data.results || [])
        onResultsChange?.(data.results || [])
      }
    } catch (error) {
      console.error('Search failed:', error)
    } finally {
      setIsSearching(false)
    }
  }

  const handleSearchSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    performSearch()
  }

  const acceptCompletion = () => {
    if (inlineCompletion) {
      setSearchQuery(inlineCompletion)
      setInlineCompletion('')

      // Auto-apply filters based on suggestion type
      const suggestion = suggestions[0]
      if (suggestion && suggestion.type === 'date') {
        if (suggestion.text.includes('7 days')) {
          setFilters(prev => ({ ...prev, dateRange: '7d' }))
        } else if (suggestion.text.includes('30 days') || suggestion.text.includes('month')) {
          setFilters(prev => ({ ...prev, dateRange: '30d' }))
        }
      }

      // Trigger search after accepting completion
      setTimeout(() => performSearch(), 100)
    }
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Tab' && inlineCompletion) {
      e.preventDefault()
      acceptCompletion()
    } else if (e.key === 'ArrowRight' && inlineCompletion) {
      // Also accept completion with right arrow if cursor is at end
      const input = inputRef.current
      if (input && input.selectionStart === searchQuery.length) {
        e.preventDefault()
        acceptCompletion()
      }
    } else if (e.key === 'Escape') {
      setInlineCompletion('')
    }
  }

  const toggleFilter = (filterType: keyof SearchFilters, value: any) => {
    setFilters(prev => {
      const newFilters = { ...prev }
      if (newFilters[filterType] === value) {
        delete newFilters[filterType]
      } else {
        newFilters[filterType] = value
      }
      return newFilters
    })

    // Trigger search after filter change
    setTimeout(() => performSearch(), 100)
  }

  const clearSearch = () => {
    setSearchQuery('')
    setFilters({})
    setSearchResults([])
    onResultsChange?.([])
  }

  return (
    <div className="w-full space-y-3">
      {/* Search Bar */}
      <div ref={searchRef} className="relative">
        <form onSubmit={handleSearchSubmit}>
          <div className="relative">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-slate-400 z-10" />

            {/* Background for input */}
            <div className="absolute inset-0 bg-slate-800 rounded-lg pointer-events-none"></div>

            {/* Inline completion overlay - positioned exactly over input text */}
            {inlineCompletion && (
              <div
                className="absolute left-0 top-0 w-full h-full flex items-center pointer-events-none z-10"
                style={{ paddingLeft: '2.5rem', paddingRight: '2.5rem' }}
              >
                <span className="text-sm text-transparent select-none">{searchQuery}</span>
                <span className="text-sm text-slate-500 select-none">{inlineCompletion.slice(searchQuery.length)}</span>
              </div>
            )}

            <input
              ref={inputRef}
              type="text"
              value={searchQuery}
              onChange={(e) => {
                setSearchQuery(e.target.value)
              }}
              onKeyDown={handleKeyDown}
              placeholder="Search emails... (e.g., 'emails from @github.com')"
              className="w-full bg-transparent border border-slate-600 rounded-lg pl-10 pr-10 py-3 text-sm text-white placeholder:text-slate-400 focus:outline-none focus:ring-2 focus:ring-violet-500 relative z-20"
            />
            {searchQuery && (
              <button
                type="button"
                onClick={clearSearch}
                className="absolute right-3 top-1/2 transform -translate-y-1/2 text-slate-400 hover:text-slate-200 z-30"
              >
                <X className="w-4 h-4" />
              </button>
            )}
          </div>
        </form>

        {/* Hint text */}
        {inlineCompletion && (
          <div className="text-xs text-slate-500 mt-1">
            Press <kbd className="px-1 py-0.5 bg-slate-700 rounded text-slate-300">Tab</kbd> or <kbd className="px-1 py-0.5 bg-slate-700 rounded text-slate-300">→</kbd> to accept
          </div>
        )}
      </div>

      {/* Filter Buttons */}
      <div className="flex items-center gap-2 flex-wrap">

        {/* Quick Filters */}
        <button
          onClick={() => toggleFilter('dateRange', '7d')}
          className={`text-xs px-3 py-1.5 rounded-full transition-colors flex items-center gap-1 ${
            filters.dateRange === '7d'
              ? 'bg-violet-600 text-white'
              : 'bg-slate-700 hover:bg-slate-600'
          }`}
        >
          <Calendar className="w-3 h-3" />
          Last 7 days
        </button>

        <button
          onClick={() => toggleFilter('dateRange', '30d')}
          className={`text-xs px-3 py-1.5 rounded-full transition-colors flex items-center gap-1 ${
            filters.dateRange === '30d'
              ? 'bg-violet-600 text-white'
              : 'bg-slate-700 hover:bg-slate-600'
          }`}
        >
          <Calendar className="w-3 h-3" />
          Last 30 days
        </button>

        <button
          onClick={() => toggleFilter('hasAttachment', true)}
          className={`text-xs px-3 py-1.5 rounded-full transition-colors flex items-center gap-1 ${
            filters.hasAttachment
              ? 'bg-violet-600 text-white'
              : 'bg-slate-700 hover:bg-slate-600'
          }`}
        >
          <Paperclip className="w-3 h-3" />
          Has attachments
        </button>

        {(searchQuery || Object.keys(filters).length > 0) && (
          <button
            onClick={clearSearch}
            className="text-xs text-slate-400 hover:text-white transition-colors ml-auto"
          >
            Clear all
          </button>
        )}
      </div>

      {/* Search Results */}
      {isSearching && (
        <div className="text-center py-4 text-slate-400 text-sm">
          <div className="flex items-center justify-center gap-2">
            <div className="w-2 h-2 bg-violet-500 rounded-full animate-pulse"></div>
            <div className="w-2 h-2 bg-violet-500 rounded-full animate-pulse" style={{animationDelay: '0.2s'}}></div>
            <div className="w-2 h-2 bg-violet-500 rounded-full animate-pulse" style={{animationDelay: '0.4s'}}></div>
          </div>
        </div>
      )}

      {!isSearching && searchResults.length > 0 && (
        <div className="space-y-2">
          <div className="text-xs text-slate-400 mb-2">
            Found {searchResults.length} email{searchResults.length > 1 ? 's' : ''}
          </div>
          {searchResults.map((email, index) => (
            <a
              key={index}
              href={`https://mail.google.com/mail/u/0/#inbox/${email.thread_id}`}
              target="_blank"
              rel="noopener noreferrer"
              className="block bg-slate-800/50 hover:bg-slate-700 border border-slate-600 rounded-lg p-3 transition-colors cursor-pointer"
            >
              <div className="flex items-start justify-between gap-2">
                <div className="flex-1 min-w-0">
                  <div className="text-sm font-medium text-slate-100 truncate">
                    {email.subject || '(No Subject)'}
                  </div>
                  <div className="text-xs text-slate-400 mt-1 truncate">
                    {email.sender}
                  </div>
                  {email.content && (
                    <div className="text-xs text-slate-500 mt-1 line-clamp-2">
                      {email.content.substring(0, 150)}...
                    </div>
                  )}
                </div>
                <div className="flex flex-col items-end gap-1 flex-shrink-0">
                  <div className="text-xs text-slate-500">
                    {new Date(email.created_at).toLocaleDateString()}
                  </div>
                  {email.has_attachment && (
                    <Paperclip className="w-3 h-3 text-slate-400" />
                  )}
                </div>
              </div>
            </a>
          ))}
        </div>
      )}

      {!isSearching && searchQuery && searchResults.length === 0 && (
        <div className="text-center py-8 text-slate-400 text-sm">
          No emails found matching your search
        </div>
      )}
    </div>
  )
}
