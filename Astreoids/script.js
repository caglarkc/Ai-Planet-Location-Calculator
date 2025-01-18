// CSV verilerini yükle ve parse et
async function loadCSVData() {
    try {
        const response = await fetch('new_data.csv');
        const data = await response.text();
        const lines = data.split('\n');
        const headers = lines[0].split(',');
        // İlk satır (başlıklar) ve sonraki 10 satırı al
        return lines.slice(1, 1001).map(line => {
            const values = line.split(',');
            return {
                H: values[0],
                M: values[1],
                Peri: values[2],
                Node: values[3],
                Incl: values[4],
                e: values[5],
                n: values[6],
                a: values[7],
                ms: values[8],
                Perts1: values[9],
                Perts2: values[10],
                name: values[11],
                epoch: values[12]
            };
        });
    } catch (error) {
        console.error('Veri yükleme hatası:', error);
        return [];
    }
}

// CSV verilerini parse et
function parseCSV(csv) {
    const lines = csv.split('\n');
    const headers = lines[0].split(',');
    return lines.slice(1).map(line => {
        const values = line.split(',');
        return {
            H: values[0],
            M: values[1],
            Peri: values[2],
            Node: values[3],
            Incl: values[4],
            e: values[5],
            n: values[6],
            a: values[7],
            ms: values[8],
            Perts1: values[9],
            Perts2: values[10],
            name: values[11],
            epoch: values[12]
        };
    });
}

// Tabloyu doldur
function populateTable(data) {
    const tbody = document.getElementById('tableBody');
    tbody.innerHTML = '';

    data.forEach(row => {
        const tr = document.createElement('tr');
        tr.innerHTML = `
            <td>${row.name}</td>
            <td>${parseFloat(row.H).toFixed(2)}</td>
            <td>${parseFloat(row.M).toFixed(2)}°</td>
            <td>${parseFloat(row.Incl).toFixed(2)}°</td>
            <td>${parseFloat(row.e).toFixed(4)}</td>
            <td>${parseFloat(row.a).toFixed(2)} AU</td>
            <td>${row.epoch}</td>
            <td><button onclick="showDetails('${row.name}')">Detaylar</button></td>
        `;
        tbody.appendChild(tr);
    });
}

// Arama fonksiyonu
function searchTable() {
    const input = document.getElementById('searchInput');
    const filter = input.value.toUpperCase();
    const tbody = document.getElementById('tableBody');
    const tr = tbody.getElementsByTagName('tr');

    for (let i = 0; i < tr.length; i++) {
        const td = tr[i].getElementsByTagName('td');
        let txtValue = '';
        for (let j = 0; j < td.length; j++) {
            txtValue += td[j].textContent || td[j].innerText;
        }
        if (txtValue.toUpperCase().indexOf(filter) > -1) {
            tr[i].style.display = '';
        } else {
            tr[i].style.display = 'none';
        }
    }
}

// Detay gösterme fonksiyonu
function showDetails(name) {
    window.location.href = `asteroid-detail.html?name=${name}`;
}

// Event listeners
document.addEventListener('DOMContentLoaded', async () => {
    const data = await loadCSVData();
    populateTable(data);

    document.getElementById('searchInput').addEventListener('keyup', searchTable);
});

// Filtreleme fonksiyonu
function applyFilter() {
    const column = document.getElementById('filterColumn').value;
    const value = document.getElementById('filterValue').value.toLowerCase();
    const tbody = document.getElementById('tableBody');
    const tr = tbody.getElementsByTagName('tr');

    for (let i = 0; i < tr.length; i++) {
        const td = tr[i].getElementsByTagName('td');
        const cellValue = td[getColumnIndex(column)].textContent.toLowerCase();
        
        if (cellValue.indexOf(value) > -1) {
            tr[i].style.display = '';
        } else {
            tr[i].style.display = 'none';
        }
    }
}

function getColumnIndex(column) {
    const columns = {
        'name': 0,
        'H': 1,
        'e': 4,
        'epoch': 6
    };
    return columns[column] || 0;
}

function resetFilters() {
    document.getElementById('filterValue').value = '';
    document.getElementById('searchInput').value = '';
    const tbody = document.getElementById('tableBody');
    const tr = tbody.getElementsByTagName('tr');
    for (let i = 0; i < tr.length; i++) {
        tr[i].style.display = '';
    }
} 