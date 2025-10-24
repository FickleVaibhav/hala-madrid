# Hardware Assembly Instructions

## Overview
This guide provides step-by-step instructions for building a smartphone-connected digital stethoscope for respiratory sound analysis. The design is inspired by the MixPose digital stethoscope but simplified for direct smartphone connection without microcontrollers.

## Safety Precautions
⚠️ **Important Safety Notes**:
- This device is for research and educational purposes only
- Not intended for clinical diagnosis or medical treatment
- Always sanitize equipment between uses
- Ensure proper electrical isolation when using amplification circuits
- Follow local regulations for medical device development

## Required Tools
- Soldering iron (15-25W)
- Solder (rosin core, 0.6-0.8mm)
- Wire strippers
- Small screwdrivers (Phillips, flathead)
- Hot glue gun or epoxy
- Multimeter (for testing connections)
- Heat shrink tubing
- 3D printer (optional, for custom housings)

## Assembly Steps

### Step 1: Stethoscope Chest Piece Preparation

#### Option A: Using Existing Stethoscope
1. **Acquire a basic stethoscope** ($5-15 from medical supply stores)
2. **Remove the tubing** by unscrewing or cutting at the chest piece connection
3. **Clean the chest piece** thoroughly with isopropyl alcohol
4. **Inspect the diaphragm** - ensure it moves freely and has good acoustic properties

#### Option B: 3D Printed Chest Piece
1. **Download STL files** from `hardware/3d_models/stethoscope_adapter.stl`
2. **Print using PLA or PETG** (0.2mm layer height, 20% infill)
3. **Sand smooth** and test fit with microphone housing
4. **Apply acoustic dampening** material inside (foam padding)

### Step 2: Piezo Microphone Preparation

#### Recommended Microphone: CM-01B Contact Microphone
**Specifications:**
- Frequency response: 20Hz - 20kHz
- Sensitivity: -42dB ± 3dB
- Operating voltage: 1.5V - 10V
- Output impedance: 2.2kΩ
- Size: 20mm diameter, 0.5mm thickness

#### Alternative: DIY Piezo Element
1. **Source piezo ceramic disc** (20-27mm diameter)
2. **Solder red wire to center electrode** (positive terminal)
3. **Solder black wire to outer ring** (ground/negative)
4. **Apply strain relief** with heat shrink tubing
5. **Test continuity** with multimeter

### Step 3: Audio Connection Cable

#### 3.5mm TRRS Connection (CTIA Standard)
**Pinout Configuration:**
- **Tip (T)**: Left audio channel
- **Ring 1 (R)**: Right audio channel  
- **Ring 2 (R)**: Ground/Shield
- **Sleeve (S)**: Microphone input

#### Cable Preparation
1. **Strip 3.5mm TRRS cable** exposing 4 conductors
2. **Identify wires** using multimeter continuity test:
   - Red/White: Audio channels (not used)
   - Black/Copper: Ground/Shield
   - Green/Blue: Microphone input
3. **Prepare wire ends** by tinning with solder

#### USB-C Alternative
For smartphones without 3.5mm jack:
1. **Use USB-C to 3.5mm adapter** with analog audio support
2. **Verify compatibility** - must support audio input, not just output
3. **Test with voice recorder app** before final assembly

### Step 4: Circuit Assembly

#### Basic Connection (No Amplification)
```
Piezo Microphone    3.5mm TRRS Plug
Red (+) ----------> Microphone Input (Ring 2)
Black (-) ---------> Ground (Sleeve)
```

#### With LM386 Amplification Circuit (Optional)
**Components needed:**
- LM386 audio amplifier IC
- 10μF capacitor (input coupling)
- 220μF capacitor (output coupling)  
- 10kΩ potentiometer (gain control)
- 0.1μF bypass capacitor
- Small PCB or breadboard

**Circuit Connections:**
```
Piezo (+) -> 10μF Cap -> LM386 Pin 3 (Non-inverting input)
LM386 Pin 2 (Inverting input) -> Ground
LM386 Pin 4 (Vs-) -> Ground  
LM386 Pin 6 (Vs+) -> 3-9V supply
LM386 Pin 5 (Output) -> 220μF Cap -> 3.5mm Microphone Input
10kΩ Pot between Pin 1 and Pin 8 for gain control
0.1μF bypass cap between Pin 7 and Ground
```

### Step 5: Physical Assembly

#### Microphone Attachment
1. **Clean chest piece diaphragm** with alcohol
2. **Apply thin layer of conductive adhesive** or use mechanical clamp
3. **Position piezo element** centered on diaphragm inner surface
4. **Ensure good acoustic coupling** without air gaps
5. **Route wires** through chest piece opening
6. **Secure with strain relief** (heat shrink or cable grip)

#### Housing and Protection
1. **3D print microphone housing** (`hardware/3d_models/microphone_housing.stl`)
2. **Install amplifier circuit** (if used) in housing
3. **Add moisture protection** with conformal coating or potting compound
4. **Label clearly** as "Research Device - Not for Clinical Use"

### Step 6: Testing and Calibration

#### Basic Functionality Test
1. **Connect to smartphone** audio input
2. **Open audio recording app** (Voice Recorder, Audacity mobile, etc.)
3. **Test microphone sensitivity**:
   - Tap chest piece gently
   - Record ambient room noise
   - Verify clear audio signal without distortion
4. **Check for electrical noise** or interference

#### Acoustic Testing
1. **Record heart sounds** (place on chest)
2. **Record lung sounds** (various chest positions)
3. **Test frequency response** using tone generator app
4. **Verify 20Hz-2000Hz response** suitable for respiratory sounds
5. **Document baseline noise level**

#### Smartphone Compatibility
**Test with multiple devices:**
- Android phones (Samsung, Google Pixel, OnePlus)
- iPhone with USB-C/Lightning adapter
- Tablets with audio input capability

**Verify specifications:**
- Sample rate: 16kHz minimum (44.1kHz preferred)
- Bit depth: 16-bit minimum
- Input sensitivity: Compatible with 2.2V mic bias
- Frequency response: Flat 20Hz-20kHz

### Step 7: Software Integration

#### Install Project Dependencies
```bash
git clone https://github.com/yourproject/respiratory-disease-detection
cd respiratory-disease-detection
pip install -r requirements.txt
```

#### Test Hardware with Streamlit App
```bash
streamlit run web_app/streamlit_app.py
```

#### Verify Audio Input Pipeline
1. **Record 10-second test sample** 
2. **Check waveform visualization**
3. **Generate spectrogram**  
4. **Test preprocessing pipeline**
5. **Run basic classification**

## Troubleshooting

### Common Issues

#### No Audio Signal
- **Check cable continuity** with multimeter
- **Verify smartphone compatibility** (test with other microphones)
- **Confirm proper TRRS pinout** (CTIA vs OMTP standards)
- **Test piezo element** by tapping and listening for output

#### Weak Signal
- **Add amplification circuit** (LM386 or similar)
- **Improve mechanical coupling** between piezo and chest piece
- **Check smartphone input gain settings**
- **Try different microphone positioning**

#### Electrical Interference
- **Add ground shielding** to cables
- **Use twisted pair wiring** for longer connections
- **Check for nearby RF sources** (WiFi, Bluetooth, cellular)
- **Add ferrite cores** to cable ends if needed

#### Poor Acoustic Quality
- **Verify chest piece seal** around diaphragm
- **Check for air leaks** in acoustic path
- **Clean contact surfaces** (remove oxidation, debris)
- **Test different chest piece materials** (metal vs plastic)

### Advanced Modifications

#### Multi-Channel Recording
- Use stereo 3.5mm connection for two microphones
- Enable simultaneous chest/back recording
- Implement spatial audio processing

#### Wireless Connection
- Add Bluetooth audio transmitter module
- Use ESP32 with built-in WiFi/Bluetooth
- Implement direct smartphone app communication

#### Signal Processing Enhancements
- Hardware-based noise reduction (active filtering)
- Real-time signal conditioning (auto-gain control)
- Multi-band compression for dynamic range

## Validation and Testing

### Acoustic Performance Standards
- **Frequency Response**: ±3dB from 20Hz-2000Hz
- **Signal-to-Noise Ratio**: >40dB for respiratory sounds
- **Dynamic Range**: >60dB for clinical applications
- **Harmonic Distortion**: <1% THD at normal levels

### Comparison with Commercial Stethoscopes
Test against:
- **Littmann Classic III** (acoustic benchmark)
- **3M Littmann 3200** (electronic reference)
- **Thinklabs One** (digital reference)

### Dataset Validation
Record standardized test sounds:
- **Normal breathing** (various positions)
- **Simulated pathological sounds** (wheeze/crackle recordings)
- **Environmental noise** (baseline measurements)
- **Calibration tones** (1kHz, 100Hz, 2kHz references)

## Maintenance and Care

### Cleaning Protocol
1. **Disconnect electronics** before cleaning
2. **Use 70% isopropyl alcohol** for chest piece
3. **Avoid moisture** on electrical connections
4. **Air dry completely** before reassembly
5. **Store in protective case** when not in use

### Regular Calibration
- **Monthly acoustic checks** with reference sounds
- **Annual electrical testing** (continuity, impedance)
- **Update software** regularly for improvements
- **Document performance** over time

### Replacement Parts
- **Piezo elements**: 6-month typical lifespan with regular use
- **Cables**: Replace if continuity issues develop
- **Chest pieces**: Clean/replace diaphragm as needed
- **Amplifier components**: Check for drift in gain/frequency response

---

**Safety Reminder**: This device is for research purposes only. Not intended for clinical diagnosis. Always consult qualified healthcare professionals for medical evaluations.