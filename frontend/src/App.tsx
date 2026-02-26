import React, { useState } from 'react';
import axios from 'axios';
import { Upload, FileText, Activity, User, Brain, AlertCircle, Image as ImageIcon, Info, ShieldAlert, Microscope, Stethoscope, ClipboardList } from 'lucide-react';

// Types
interface AnalysisResult {
  original_image: string;
  segmented_image: string;
  metrics: { "Volume (Approx.)": string; "Max Diameter (Approx.)": string };
  confidence: number;
  stage: string;
  location: string;
}

function App() {
  const [activeTab, setActiveTab] = useState<'intake' | 'imaging' | 'report'>('intake');
  const [file, setFile] = useState<File | null>(null);
  const [patient, setPatient] = useState({ name: '', age: '', contact: '' });
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) setFile(e.target.files[0]);
  };

  const handleAnalyze = async () => {
    if (!file || !patient.name) {
      setError("Please provide patient details and an MRI scan.");
      return;
    }
    setLoading(true);
    setError('');
    
    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await axios.post('https://neuroai-backend-8lim.onrender.com/analyze', formData);
      setResult(response.data);
      setActiveTab('imaging'); 
    } catch (err) {
      setError("Analysis failed. Ensure backend is running.");
    } finally {
      setLoading(false);
    }
  };

  const downloadReport = async () => {
    if (!result) return;
    const formData = new FormData();
    formData.append('name', patient.name);
    formData.append('age', patient.age);
    formData.append('contact', patient.contact);
    formData.append('original', result.original_image);
    formData.append('segmented', result.segmented_image);
    formData.append('volume', result.metrics['Volume (Approx.)']);
    formData.append('diameter', result.metrics['Max Diameter (Approx.)']);
    formData.append('stage', result.stage);
    formData.append('confidence', result.confidence.toString());
    formData.append('location', result.location);

    const response = await axios.post('https://neuroai-backend-8lim.onrender.com/report', formData, { responseType: 'blob' });
    const url = window.URL.createObjectURL(new Blob([response.data]));
    const link = document.createElement('a');
    link.href = url;
    link.setAttribute('download', `Report_${patient.name}.pdf`);
    document.body.appendChild(link);
    link.click();
  };

  // DYNAMIC TREATMENT PLAN GENERATOR
  const getTreatmentPlan = (stage: string) => {
    if (stage.includes('IV')) {
        return {
            title: "Urgent Aggressive Intervention (High Grade)",
            badgeColor: "bg-red-100 text-red-700",
            iconColor: "text-red-600",
            steps: [
                "Immediate referral to a multidisciplinary Neuro-Oncology tumor board.",
                "Urgent Surgical Intervention: Goal is Maximum Safe Resection (MSR) guided by intraoperative neuronavigation.",
                "Adjuvant Therapy: Prepare for standard concurrent chemoradiation (e.g., Stupp Protocol - Temozolomide + Radiotherapy) post-surgery.",
                "Molecular Profiling: Tissue biopsy should be tested for MGMT promoter methylation status to guide targeted therapy."
            ]
        };
    } else if (stage.includes('III')) {
        return {
            title: "Standard Adjuvant Protocol (Intermediate Grade)",
            badgeColor: "bg-orange-100 text-orange-700",
            iconColor: "text-orange-600",
            steps: [
                "Referral to Neuro-Oncology for comprehensive case review and surgical planning.",
                "Surgical Resection: Aim for gross total resection where anatomically feasible without inducing severe neurological deficit.",
                "Post-Surgical Therapy: High likelihood of requiring adjuvant fractionated radiotherapy combined with systemic chemotherapy.",
                "Monitoring: High-frequency MRI monitoring (every 2-3 months) post-treatment to monitor for malignant transformation."
            ]
        };
    } else {
        return {
            title: "Conservative Management (Low Grade)",
            badgeColor: "bg-green-100 text-green-700",
            iconColor: "text-green-600",
            steps: [
                "Histological Verification: Stereotactic biopsy or safe resection to confirm low-grade status.",
                "Active Surveillance: If asymptomatic and completely resected, 'watch and wait' with serial MRI scans (every 3-6 months).",
                "Symptom Management: If symptomatic (e.g., seizures), focal resection is the primary treatment.",
                "Therapy Deferment: Radiotherapy/Chemotherapy is generally deferred unless the tumor shows signs of rapid progression or is inoperable."
            ]
        };
    }
  };

  const getTabClass = (tabName: string) => {
    return activeTab === tabName 
      ? "flex items-center gap-3 px-4 py-3 bg-indigo-50 text-primary font-medium rounded-lg cursor-pointer transition-colors"
      : "flex items-center gap-3 px-4 py-3 text-gray-500 hover:bg-gray-50 rounded-lg cursor-pointer transition-colors";
  };

  return (
    <div className="flex min-h-screen bg-gray-50 text-gray-800 font-sans">
      
      {/* SIDEBAR NAVIGATION */}
      <aside className="w-72 bg-white border-r border-gray-200 hidden md:flex flex-col justify-between">
        <div>
            <div className="p-6 border-b border-gray-100">
            <h1 className="text-xl font-bold text-primary flex items-center gap-2">
                <Brain className="w-8 h-8" /> NeuroAI
            </h1>
            <p className="text-xs text-gray-400 mt-1">Clinical Decision Support System</p>
            </div>
            
            <nav className="p-4 space-y-2">
            <div className={getTabClass('intake')} onClick={() => setActiveTab('intake')}>
                <User size={20} /> Patient Intake
            </div>
            <div 
                className={`${getTabClass('imaging')} ${!result ? 'opacity-50 cursor-not-allowed' : ''}`}
                onClick={() => result && setActiveTab('imaging')}
            >
                <ImageIcon size={20} /> Imaging & Metrics
            </div>
            <div 
                className={`${getTabClass('report')} ${!result ? 'opacity-50 cursor-not-allowed' : ''}`}
                onClick={() => result && setActiveTab('report')}
            >
                <Activity size={20} /> Clinical Report
            </div>
            </nav>
        </div>

        {/* STATIC CONTEXT: Sidebar System Status */}
        <div className="p-6 bg-gray-50 m-4 rounded-xl border border-gray-200">
            <h4 className="text-xs font-bold text-gray-500 uppercase tracking-wider mb-3 flex items-center gap-2">
                <ShieldAlert size={14}/> System Status
            </h4>
            <div className="space-y-2 text-xs text-gray-600">
                <p className="flex justify-between"><span>Model:</span> <span className="font-medium">Hybrid U-Net v2.1</span></p>
                <p className="flex justify-between"><span>Backend:</span> <span className="text-green-600 font-medium text-right">Online (FastAPI)</span></p>
                <p className="flex justify-between"><span>Input Type:</span> <span className="font-medium">2D MRI Slices</span></p>
                <p className="flex justify-between"><span>Validation:</span> <span className="font-medium">Research Only</span></p>
            </div>
        </div>
      </aside>

      {/* MAIN CONTENT AREA */}
      <main className="flex-1 p-8 overflow-y-auto">
        <header className="flex justify-between items-center mb-8">
          <h2 className="text-2xl font-bold text-gray-800">
            {activeTab === 'intake' && "New Analysis Session"}
            {activeTab === 'imaging' && "Radiological Imaging"}
            {activeTab === 'report' && "Diagnostic Summary"}
          </h2>
          <div className="flex items-center gap-3 bg-white px-4 py-2 rounded-full shadow-sm border">
            <div className="w-8 h-8 bg-indigo-100 rounded-full flex items-center justify-center text-primary font-bold">
              Dr
            </div>
            <span className="text-sm font-medium">Radiology Dept.</span>
          </div>
        </header>

        {error && <div className="bg-red-50 text-red-600 p-4 rounded-lg mb-6 flex items-center gap-2"><AlertCircle size={20}/>{error}</div>}

        {/* VIEW 1: PATIENT INTAKE */}
        {activeTab === 'intake' && (
          <div className="flex flex-col gap-6 animate-fade-in">
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-100 col-span-2">
                <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
                    <User size={20} className="text-indigo-500"/> Patient Demographics
                </h3>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    <input type="text" placeholder="Full Name" className="border p-3 rounded-lg w-full bg-gray-50 focus:outline-none focus:ring-2 focus:ring-primary/50" 
                    value={patient.name} onChange={e => setPatient({...patient, name: e.target.value})} />
                    <input type="text" placeholder="Age" className="border p-3 rounded-lg w-full bg-gray-50 focus:outline-none focus:ring-2 focus:ring-primary/50" 
                    value={patient.age} onChange={e => setPatient({...patient, age: e.target.value})} />
                    <input type="text" placeholder="ID / Contact" className="border p-3 rounded-lg w-full bg-gray-50 focus:outline-none focus:ring-2 focus:ring-primary/50" 
                    value={patient.contact} onChange={e => setPatient({...patient, contact: e.target.value})} />
                </div>
                </div>

                <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-100 flex flex-col items-center justify-center text-center border-dashed border-2 border-indigo-100 hover:border-indigo-300 transition-colors">
                <Upload className="w-10 h-10 text-indigo-400 mb-2" />
                <label className="cursor-pointer">
                    <span className="text-primary font-semibold hover:underline">Click to Upload MRI</span>
                    <input type="file" className="hidden" onChange={handleFileChange} accept="image/*" />
                </label>
                <p className="text-xs text-gray-400 mt-1">{file ? file.name : "Supported: PNG, JPG, JPEG"}</p>
                <button 
                    onClick={handleAnalyze} 
                    disabled={loading}
                    className={`mt-4 w-full py-2 rounded-lg font-medium text-white transition-all ${loading ? 'bg-gray-400' : 'bg-primary hover:bg-indigo-600 shadow-lg shadow-indigo-200'}`}
                >
                    {loading ? "Processing Segmentation..." : "Run AI Segmentation"}
                </button>
                </div>
            </div>

            {/* STATIC CONTEXT: Intake SOP */}
            <div className="bg-blue-50 border border-blue-100 p-6 rounded-xl flex gap-4 items-start">
                <Info className="text-blue-500 shrink-0 w-6 h-6 mt-1"/>
                <div>
                    <h4 className="font-semibold text-blue-900 mb-1">Standard Operating Procedure (SOP)</h4>
                    <p className="text-sm text-blue-800/80 leading-relaxed">
                        This AI-assisted segmentation tool is optimized for <strong>T1-weighted (post-contrast), T2-weighted, and FLAIR MRI slices</strong>. 
                        Ensure the uploaded image is a clear, single 2D axial slice. The model utilizes a Hybrid Watershed + U-Net architecture to 
                        minimize false positives. <em>Note: This software provides secondary clinical decision support and does not replace professional histological diagnosis.</em>
                    </p>
                </div>
            </div>
          </div>
        )}

        {/* VIEW 2: IMAGING & METRICS */}
        {activeTab === 'imaging' && result && (
          <div className="space-y-6 animate-fade-in">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="bg-white p-5 rounded-xl border border-gray-100 shadow-sm border-l-4 border-l-blue-500">
                <p className="text-xs text-gray-500 uppercase font-semibold">AI Confidence</p>
                <p className={`text-3xl font-bold mt-1 ${result.confidence > 80 ? 'text-green-500' : 'text-orange-500'}`}>
                  {result.confidence.toFixed(2)}%
                </p>
              </div>
              <div className="bg-white p-5 rounded-xl border border-gray-100 shadow-sm border-l-4 border-l-purple-500">
                <p className="text-xs text-gray-500 uppercase font-semibold">Tumor Volume</p>
                <p className="text-2xl font-bold mt-1 text-gray-800">{result.metrics["Volume (Approx.)"]}</p>
              </div>
               <div className="bg-white p-5 rounded-xl border border-gray-100 shadow-sm border-l-4 border-l-indigo-500">
                <p className="text-xs text-gray-500 uppercase font-semibold">Max Diameter</p>
                <p className="text-2xl font-bold mt-1 text-gray-800">{result.metrics["Max Diameter (Approx.)"]}</p>
              </div>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-100 col-span-2">
                    <h4 className="font-semibold mb-4 text-gray-700 flex items-center gap-2"><ImageIcon size={20} className="text-primary"/> Segmentation Mask Overlay</h4>
                    <div className="relative aspect-video bg-gray-900 rounded-lg overflow-hidden flex justify-center items-center">
                        <img src={result.segmented_image} alt="Segmented" className="object-contain h-full w-full" />
                    </div>
                    <div className="mt-4 flex items-center justify-center gap-4 text-sm text-gray-500">
                        <span className="flex items-center gap-1"><div className="w-3 h-3 bg-red-500 rounded-full"></div> AI Tumor Boundary</span>
                        <span className="flex items-center gap-1"><div className="w-3 h-3 bg-gray-400 rounded-full"></div> Healthy Tissue</span>
                    </div>
                </div>

                {/* STATIC CONTEXT: Metric Definitions */}
                <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-100 flex flex-col gap-4">
                    <h4 className="font-semibold text-gray-700 flex items-center gap-2 border-b pb-3"><Microscope size={20} className="text-indigo-500"/> Clinical Relevance</h4>
                    
                    <div className="text-sm">
                        <span className="font-bold text-gray-800 block mb-1">Volumetric Analysis</span>
                        <p className="text-gray-600 mb-3">Volume calculations are derived from pixel-to-millimeter ratio conversions. Large volumes (&gt;40cm³) often correlate with higher intracranial pressure and higher WHO grading.</p>
                        
                        <span className="font-bold text-gray-800 block mb-1">Surgical Margins</span>
                        <p className="text-gray-600">The delineated red boundary is critical for planning <strong>Maximum Safe Resection (MSR)</strong>, balancing complete tumor removal with the preservation of adjacent healthy neurological tissue.</p>
                    </div>
                </div>
            </div>
          </div>
        )}

        {/* VIEW 3: CLINICAL REPORT */}
        {activeTab === 'report' && result && (
          <div className="flex flex-col gap-6 animate-fade-in">
              {/* TOP ROW: Diagnostic Summary & Export */}
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-100 md:col-span-2">
                    <h4 className="font-semibold mb-4 text-gray-700">Diagnostic Summary</h4>
                    <div className="space-y-4">
                    <div className="bg-purple-50 p-4 rounded-lg">
                        <p className="text-xs font-bold text-purple-600 uppercase mb-1">Estimated Stage</p>
                        <p className="text-lg font-bold text-gray-800">{result.stage.split('(')[0]}</p>
                        <p className="text-sm text-gray-600 mt-1">{result.stage.includes('IV') ? "High-grade pathology suspected." : "Low to intermediate grade morphology."}</p>
                    </div>

                    <div className="bg-blue-50 p-4 rounded-lg border border-blue-100">
                        <p className="text-xs font-bold text-blue-500 uppercase mb-1">Anatomical Localization</p>
                        <p className="text-md font-medium text-gray-800">{result.location}</p>
                    </div>
                    </div>
                </div>

                <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-100 flex flex-col justify-center items-center text-center">
                    <FileText className="w-12 h-12 text-indigo-300 mb-3" />
                    <h3 className="text-lg font-bold text-gray-800 mb-2">Chart Export</h3>
                    <p className="text-gray-500 text-xs mb-4">
                        Download comprehensive PDF report for patient charting.
                    </p>
                    <button 
                        onClick={downloadReport} 
                        className="flex items-center gap-2 bg-primary text-white font-medium hover:bg-indigo-600 px-5 py-2.5 rounded-lg transition-colors shadow-md w-full justify-center"
                    >
                        <FileText size={18}/> Download PDF
                    </button>
                </div>
              </div>

              {/* BOTTOM ROW: Treatment Plan & Anatomy Map */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                 {/* DYNAMIC CONTENT: Treatment Planning */}
                 <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-100 flex flex-col h-full">
                    <div className="flex items-center justify-between border-b pb-3 mb-4">
                        <h4 className="font-semibold text-gray-700 flex items-center gap-2">
                            <Stethoscope size={20} className="text-indigo-500"/> Clinical Pathway Guide
                        </h4>
                        <span className={`text-xs font-bold px-3 py-1 rounded-full ${getTreatmentPlan(result.stage).badgeColor}`}>
                            {getTreatmentPlan(result.stage).title}
                        </span>
                    </div>
                    
                    <ul className="space-y-4 flex-1">
                        {getTreatmentPlan(result.stage).steps.map((step, index) => (
                            <li key={index} className="flex gap-3 items-start">
                                <ClipboardList size={18} className={`shrink-0 mt-0.5 ${getTreatmentPlan(result.stage).iconColor}`} />
                                <span className="text-sm text-gray-700 leading-relaxed">{step}</span>
                            </li>
                        ))}
                    </ul>
                 </div>

                 {/* STATIC CONTEXT: Brain Function Map */}
                 <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-100">
                    <h4 className="font-semibold text-gray-700 border-b pb-3 mb-4">Neuroanatomical Impact Guide</h4>
                    <div className="grid grid-cols-1 gap-y-3 text-sm">
                        <div className="p-3 bg-pink-50 rounded-lg border border-pink-100 flex flex-col">
                            <span className="font-bold text-pink-700">Frontal Lobe</span>
                            <span className="text-gray-600 text-xs mt-1">Motor control, personality, problem-solving, speech (Broca's).</span>
                        </div>
                        <div className="p-3 bg-yellow-50 rounded-lg border border-yellow-100 flex flex-col">
                            <span className="font-bold text-yellow-700">Parietal Lobe</span>
                            <span className="text-gray-600 text-xs mt-1">Touch perception, body orientation, sensory discrimination.</span>
                        </div>
                        <div className="p-3 bg-green-50 rounded-lg border border-green-100 flex flex-col">
                            <span className="font-bold text-green-700">Temporal Lobe</span>
                            <span className="text-gray-600 text-xs mt-1">Auditory processing, memory, speech comprehension (Wernicke's).</span>
                        </div>
                        <div className="p-3 bg-orange-50 rounded-lg border border-orange-100 flex flex-col">
                            <span className="font-bold text-orange-700">Occipital Lobe</span>
                            <span className="text-gray-600 text-xs mt-1">Visual reception and visual interpretation.</span>
                        </div>
                    </div>
                 </div>
              </div>
          </div>
        )}

      </main>
    </div>
  );
}

export default App;