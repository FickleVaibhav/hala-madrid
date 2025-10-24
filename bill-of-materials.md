# Bill of Materials (BOM)

## Overview
This document provides a comprehensive bill of materials for building a smartphone-connected digital stethoscope for respiratory disease detection. The design focuses on affordability, accessibility, and ease of assembly while maintaining sufficient audio quality for machine learning analysis.

## Cost Summary
- **Minimum Configuration**: $8-15 USD
- **Recommended Configuration**: $15-25 USD  
- **Enhanced Configuration**: $25-40 USD

## Primary Components

### 1. Stethoscope Chest Piece
**Option A: Repurposed Medical Stethoscope**
- **Item**: Basic acoustic stethoscope
- **Specifications**: Standard chest piece (bell/diaphragm), removable tubing
- **Sources**: Medical supply stores, Amazon, eBay
- **Cost**: $5-15 USD
- **Part Numbers**: 
  - Generic dual-head stethoscope
  - Prestige Medical Clinical I Stethoscope
- **Notes**: Remove tubing, keep only chest piece assembly

**Option B: 3D Printed Chest Piece**
- **Item**: Custom 3D printed stethoscope adapter
- **Material**: PLA or PETG plastic (food-safe recommended)
- **Printing Cost**: $2-5 USD (material + electricity)
- **Files**: Provided in `hardware/3d_models/stethoscope_adapter.stl`
- **Print Settings**: 0.2mm layer height, 20% infill, supports needed
- **Post-processing**: Sand smooth, apply acoustic dampening foam

### 2. Contact Microphone (Primary Component)
**Option A: CM-01B Piezoelectric Contact Microphone (Recommended)**
- **Item**: CM-01B Contact Microphone
- **Specifications**:
  - Frequency Response: 20Hz - 20kHz
  - Sensitivity: -42dB ± 3dB  
  - Operating Voltage: 1.5V - 10V
  - Output Impedance: 2.2kΩ
  - Diameter: 20mm, Thickness: 0.5mm
- **Sources**: Electronics suppliers, online marketplaces
- **Cost**: $3-8 USD
- **Part Numbers**: CM-01B, equivalent contact microphones
- **Advantages**: Pre-amplified, good sensitivity, robust design

**Option B: DIY Piezo Ceramic Disc**
- **Item**: Piezo ceramic disc + wiring
- **Specifications**: 20-27mm diameter, brass/ceramic construction
- **Additional Items Needed**:
  - Thin gauge wire (26-30 AWG) - $1-2
  - Solder and flux - $2-3
  - Heat shrink tubing - $1
- **Sources**: Electronic component suppliers (Mouser, DigiKey, local)
- **Cost**: $2-5 USD total
- **Advantages**: Lower cost, customizable
- **Disadvantages**: Requires more assembly skill, lower sensitivity

### 3. Audio Connection Cable
**Option A: 3.5mm TRRS Cable (Most Compatible)**
- **Item**: 3.5mm TRRS (4-conductor) cable
- **Specifications**: 
  - CTIA standard pinout (L-R-G-M)
  - Length: 1-3 feet (0.3-1m)
  - Conductor gauge: 26-28 AWG
- **Sources**: Electronics stores, online retailers
- **Cost**: $2-5 USD
- **Part Numbers**: 
  - Generic 3.5mm TRRS cable
  - Hosa CMR-206 (6 ft)
  - StarTech MUYHSMFF (6 ft)
- **Compatibility**: Most smartphones, tablets, computers with headphone jack

**Option B: USB-C Audio Adapter + Cable**
- **Item**: USB-C to 3.5mm adapter with analog audio support
- **Specifications**: Must support audio input (not just output)
- **Cost**: $3-8 USD
- **Compatible Devices**: Modern Android phones, some tablets
- **Part Numbers**:
  - Google USB-C to 3.5mm Adapter
  - Samsung USB-C to 3.5mm Adapter  
- **Notes**: Verify input capability before purchase

**Option C: Lightning Audio Adapter (iOS)**
- **Item**: Lightning to 3.5mm adapter (Apple or MFi certified)
- **Cost**: $8-15 USD
- **Compatibility**: iPhone, iPad with Lightning port
- **Limitations**: May require additional impedance matching

### 4. Optional Signal Amplification Circuit
**Low-Noise Audio Amplifier (Recommended for weak signals)**
- **Main IC**: LM386 Audio Amplifier
- **Supporting Components**:
  - 10μF electrolytic capacitor (input coupling) - $0.10
  - 220μF electrolytic capacitor (output coupling) - $0.15
  - 10kΩ linear potentiometer (gain control) - $1.00
  - 0.1μF ceramic capacitor (bypass) - $0.05
  - Small PCB or breadboard - $1-2
- **Total Cost**: $3-5 USD
- **Sources**: Electronics component suppliers
- **Advantages**: Improves signal strength, reduces noise
- **Assembly**: Basic soldering skills required

### 5. Housing and Mechanical Components
**Protective Housing**
- **Option A**: 3D Printed Enclosure
  - Files: `hardware/3d_models/microphone_housing.stl`
  - Material cost: $1-3 USD
  - Custom fit for specific microphone types
  
- **Option B**: Small Project Box
  - Hammond 1551 series or equivalent
  - Cost: $3-6 USD
  - Requires drilling for cable entry

**Mounting Hardware**
- **Cable strain relief**: Heat shrink tubing, cable boots - $1-2
- **Mechanical fasteners**: Small screws (M2 or M3) - $1
- **Adhesive**: Conductive adhesive or double-sided tape - $1-2
- **Acoustic dampening**: Small foam pads - $1-2

### 6. Assembly Tools and Consumables
**Essential Tools**
- Soldering iron (15-25W) - $10-30 (if not owned)
- Solder (60/40 rosin core, 0.6mm) - $5-10
- Wire strippers - $8-15
- Small screwdrivers - $5-10
- Multimeter (basic) - $15-25

**Consumables**
- Heat shrink tubing assortment - $3-5
- Electrical tape - $1-2
- Isopropyl alcohol (cleaning) - $2-3
- Flux (for soldering) - $3-5

## Configuration Options

### Minimum Configuration ($8-15)
- Basic stethoscope chest piece: $5-8
- DIY piezo element: $2-3  
- 3.5mm TRRS cable: $2-4
- Basic assembly materials: $1-2
- **Total**: $10-17

### Recommended Configuration ($15-25)
- Quality stethoscope chest piece: $8-12
- CM-01B contact microphone: $5-8
- Quality 3.5mm TRRS cable: $3-5
- 3D printed housing: $2-3
- Assembly materials and tools: $2-5
- **Total**: $20-33

### Enhanced Configuration ($25-40)
- Professional stethoscope chest piece: $10-15
- CM-01B contact microphone: $5-8
- LM386 amplification circuit: $3-5
- USB-C adapter compatibility: $3-8
- Professional housing and assembly: $5-10
- **Total**: $26-46

## Vendor Information

### Electronics Components
**Online Suppliers:**
- **Mouser Electronics** (mouser.com) - Professional components, worldwide shipping
- **DigiKey** (digikey.com) - Extensive catalog, technical support
- **Adafruit** (adafruit.com) - Maker-friendly, good tutorials
- **SparkFun** (sparkfun.com) - Educational focus, quality components

**Local Sources:**
- Electronics hobby shops
- University surplus stores
- Makerspace/hackerspace component libraries

### Medical Equipment
**Stethoscope Sources:**
- **Amazon Medical Supplies** - Wide selection, competitive pricing
- **Medical supply stores** - Local availability, can inspect quality
- **eBay** - Used/surplus equipment, cost-effective options
- **Nursing/medical school bookstores** - Student-grade equipment

### 3D Printing Services
**If no printer available:**
- **Local libraries** - Many offer 3D printing services ($2-5)
- **Makerspaces** - Community access to professional printers
- **Online services** - Shapeways, Craftcloud (higher cost but professional)
- **Universities** - Often have public access programs

## Quality Considerations

### Component Selection Criteria
**Microphone Quality:**
- Frequency response: Must cover 20Hz-2000Hz minimum
- Sensitivity: Higher sensitivity reduces need for amplification
- Impedance: Should match smartphone input (typically 1-10kΩ)
- Durability: Medical-grade preferred for repeated sanitization

**Cable Quality:**
- Conductor material: Copper (not aluminum)
- Shielding: Braided or foil shield for noise reduction
- Connectors: Gold-plated contacts for reliability
- Strain relief: Proper molded boots to prevent breakage

**Construction Quality:**
- Solder joints: Clean, shiny, mechanically sound
- Wire routing: Avoid sharp bends, provide strain relief
- Housing: Adequate protection, easy access for maintenance
- Labeling: Clear identification as research device

## Cost Optimization Strategies

### Volume Purchasing
- **Student groups**: Coordinate purchases for bulk discounts
- **Maker communities**: Group buys for expensive components
- **Academic institutions**: Educational pricing often available

### Alternative Sourcing
- **Salvage electronics**: Old audio equipment for amplifier circuits
- **Surplus stores**: Government/industrial surplus for housings
- **Repair shops**: Damaged stethoscopes for chest piece salvage

### DIY Alternatives
- **PCB fabrication**: Design custom amplifier PCBs for volume production
- **3D printing optimization**: Design for minimal material usage
- **Cable assembly**: Make custom length cables from bulk wire

## Testing and Validation

### Component Testing
- **Continuity testing**: Verify all connections with multimeter
- **Impedance measurement**: Confirm microphone specifications
- **Frequency response**: Test with audio generator if available
- **Noise floor**: Measure background noise levels

### System Integration Testing
- **Smartphone compatibility**: Test with multiple devices/OS versions
- **Audio quality**: Record and analyze test signals
- **Mechanical durability**: Stress test connections and housing
- **Repeatability**: Ensure consistent performance across units

## Maintenance and Lifecycle

### Consumable Components (Regular Replacement)
- **Piezo elements**: 6-12 months typical lifespan
- **Cables**: Replace if intermittent connection issues
- **Adhesives**: Reapply as needed for secure mounting

### Upgrade Path
- **Better microphones**: Upgrade to higher-quality contact mics
- **Digital conversion**: Add ADC for direct digital output
- **Wireless capability**: Bluetooth or WiFi transmission
- **Multi-channel**: Record from multiple chest positions simultaneously

### End-of-Life Considerations
- **Component recycling**: Electronic waste disposal guidelines
- **Housing reuse**: 3D printed parts can often be recycled
- **Documentation**: Keep build logs for future reference

---

**Note**: Prices are approximate and may vary by region, supplier, and market conditions. Always verify current pricing and availability before ordering. Consider local regulations regarding medical device components and research equipment.