const SupplyChain = artifacts.require('./contracts/SupplyChain')

contract('SupplyChain', function(accounts) {
    before(async () => {
        supplyChain = await SupplyChain.deployed()
    })
    
    describe('SupplyChain functions correctly', () => {
        before("See accounts", async () => {
            console.log("ganache-cli accounts used here...")
            console.log("Accounts :", accounts)
            console.log("Contract Owner: accounts[0] ", accounts[0])
            console.log("Farmer: accounts[1] ", accounts[1])
            console.log("Manufacturer: accounts[2] ", accounts[2])
            console.log("Distributor: accounts[3] ", accounts[3])
            console.log("Retailer: accounts[4] ", accounts[4])
            console.log("Consumer: accounts[5] ", accounts[5])
        })

        it('Can add Farmer Role', async () => {
            await supplyChain.addFarmerRole(accounts[1])
            const isFarmer1 = await supplyChain.hasFarmerRole(accounts[1])
            const isFarmer2 = await supplyChain.hasFarmerRole(accounts[0])
            assert.equal(isFarmer1, true, 'Error: Should be Farmer')
            assert.equal(isFarmer2, false, 'Error: Should not be Farmer')
        })

        

        it('Can add Manufacturer Role', async () => {
            await supplyChain.addManufacturerRole(accounts[2])
            const isManufacturer1 = await supplyChain.hasManufacturerRole(accounts[2])
            const isManufacturer2 = await supplyChain.hasManufacturerRole(accounts[0])
            assert.equal(isManufacturer1, true, 'Error: Should be Manufacturer')
            assert.equal(isManufacturer2, false, 'Error: Should not be Manufacturer')
        })
        

        it('Can add Distributor Role', async () => {
            await supplyChain.addDistributorRole(accounts[3])
            const isDistributor1 = await supplyChain.hasDistributorRole(accounts[3])
            const isDistributor2 = await supplyChain.hasDistributorRole(accounts[0])
            assert.equal(isDistributor1, true, 'Error: Should be Distributor')
            assert.equal(isDistributor2, false, 'Error: Should not be Distributor')
        })
        
        

        it('Can add Retailer Role', async () => {
            await supplyChain.addRetailerRole(accounts[4])
            const isRetailer1 = await supplyChain.hasRetailerRole(accounts[4])
            const isRetailer2 = await supplyChain.hasRetailerRole(accounts[0])
            assert.equal(isRetailer1, true, 'Error: Should be Retailer')
            assert.equal(isRetailer2, false, 'Error: Should not be Retailer')
        })

        it('Can add Consumer Role', async () => {
            await supplyChain.addConsumerRole(accounts[5])
            const isConsumer1 = await supplyChain.hasConsumerRole(accounts[5])
            const isConsumer2 = await supplyChain.hasConsumerRole(accounts[0])
            assert.equal(isConsumer1, true, 'Error: Should be Consumer')
            assert.equal(isConsumer2, false, 'Error: Should not be Consumer')
        })
        

        it('Can revoke Farmer Role', async () => {
            await supplyChain.revokeFarmerRole(accounts[1])
            const isFarmer1 = await supplyChain.hasFarmerRole(accounts[1])
            assert.equal(isFarmer1, false, 'Error: Should not be Farmer')
        })


        it('Can revoke Manufacturer Role', async () => {
            await supplyChain.revokeManufacturerRole(accounts[2])
            const isManufacturer1 = await supplyChain.hasManufacturerRole(accounts[2])
            assert.equal(isManufacturer1, false, 'Error: Should not be Manufacturer')
        })


        it('Can revoke Distributor Role', async () => {
            await supplyChain.revokeDistributorRole(accounts[3])
            const isDistributor1 = await supplyChain.hasDistributorRole(accounts[3])
            assert.equal(isDistributor1, false, 'Error: Should not be Distributor')
        })

                
        it('Can revoke Retailer Role', async () => {
            await supplyChain.revokeRetailerRole(accounts[4])
            const isRetailer1 = await supplyChain.hasRetailerRole(accounts[4])
            assert.equal(isRetailer1, false, 'Error: Should not be Retailer')
        })

        it('Can revoke Consumer Role', async () => {
            await supplyChain.revokeConsumerRole(accounts[5])
            const isConsumer1 = await supplyChain.hasConsumerRole(accounts[5])
            assert.equal(isConsumer1, false, 'Error: Should not be Consumer')
        })

    })
});