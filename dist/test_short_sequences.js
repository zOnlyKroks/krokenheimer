import { BlastApiClient } from './dist/services/BlastApiClient.js';
async function testShortSequences() {
    const client = new BlastApiClient();
    // Test sequences under 14 nucleotides
    const shortSequences = [
        { raw: 'ATCG', cleaned: 'ATCG' }, // 4 nucleotides
        { raw: 'ATCGATCG', cleaned: 'ATCGATCG' }, // 8 nucleotides
        { raw: 'ATCGATCGAAA', cleaned: 'ATCGATCGAAA' }, // 11 nucleotides
        { raw: 'ATCGATCGAAATT', cleaned: 'ATCGATCGAAATT' }, // 13 nucleotides
    ];
    console.log('Testing short sequences with new fallback system...\n');
    for (let i = 0; i < shortSequences.length; i++) {
        const seq = shortSequences[i];
        console.log(`\n=== Test ${i + 1}: Sequence "${seq.raw}" (${seq.raw.length} bp) ===`);
        try {
            const startTime = Date.now();
            const results = await client.analyzeSequence(seq);
            const endTime = Date.now();
            console.log(`✅ Analysis completed in ${endTime - startTime}ms`);
            console.log(`Database: ${results.database}`);
            console.log(`Program: ${results.program}`);
            console.log(`Found ${results.hits.length} hits:`);
            results.hits.forEach((hit, index) => {
                console.log(`  ${index + 1}. ${hit.scientificName} (${hit.commonName})`);
                console.log(`     Description: ${hit.description}`);
                console.log(`     Identity: ${hit.identity.toFixed(1)}%, E-value: ${hit.eValue}`);
                console.log(`     Accession: ${hit.accession}`);
            });
        }
        catch (error) {
            console.log(`❌ Error: ${error.message}`);
        }
    }
}
// Test with very extreme cases
async function testExtremeShortSequences() {
    const client = new BlastApiClient();
    console.log('\n\n=== EXTREME SHORT SEQUENCES ===');
    const extremeSequences = [
        { raw: 'A', cleaned: 'A' }, // 1 nucleotide
        { raw: 'AT', cleaned: 'AT' }, // 2 nucleotides
        { raw: 'GGG', cleaned: 'GGG' }, // 3 nucleotides (high GC)
        { raw: 'TTTT', cleaned: 'TTTT' }, // 4 nucleotides (low GC)
    ];
    for (let i = 0; i < extremeSequences.length; i++) {
        const seq = extremeSequences[i];
        console.log(`\n--- Extreme Test ${i + 1}: "${seq.raw}" (${seq.raw.length} bp) ---`);
        try {
            const results = await client.analyzeSequence(seq);
            console.log(`✅ Success! Database: ${results.database}, ${results.hits.length} hits`);
            console.log(`First hit: ${results.hits[0]?.scientificName || 'None'}`);
        }
        catch (error) {
            console.log(`❌ Failed: ${error.message}`);
        }
    }
}
// Run tests
console.log('🧬 Starting short sequence analysis tests...');
testShortSequences()
    .then(() => testExtremeShortSequences())
    .then(() => {
    console.log('\n✨ All tests completed!');
    console.log('The system should now find "something every time for the meme" 🎉');
})
    .catch((error) => {
    console.error('Test suite failed:', error);
});
//# sourceMappingURL=test_short_sequences.js.map